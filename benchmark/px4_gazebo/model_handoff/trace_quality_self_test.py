"""Synthetic timing faults: no flight or external runtime required."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from .execution_trace import ExecutionTrace, TIMED_TRACE_SCHEMA, validate_trace, host_clock_pair
from .dataset import EpisodeDatasetWriter
from .fixtures import generate_fixtures
from .self_test import _observation
from .interface_self_test import _odometry, _reject
from .protocol import AGENT_IDS
from .trace_quality import quality_report, clock_pair_diagnostics, report_episode
from .sampling import TraceSamplingGate


def _sampling_tests(root):
    writer = EpisodeDatasetWriter(root, episode_id="strict-pairs", fixture=generate_fixtures()[0].to_dict(),
                                  mode="oracle", record_execution=True, strict_image_pairs=True)
    obs = replace(_observation(), episode_id="strict-pairs")
    from .protocol import ModelJointAction
    writer.append(obs, ModelJointAction(), events=[], truth={}, protocol_t=0)
    next_stamps = {agent: stamp + 1 for agent, stamp in obs.image_timestamp_ns.items()}
    next_stamps["uav_0"] = obs.image_timestamp_ns["uav_0"]
    before = writer.samples_path.read_bytes()
    _reject(lambda: writer.append(replace(obs, timestamp_ns=max(next_stamps.values()), image_timestamp_ns=next_stamps),
                                  ModelJointAction(), events=[], truth={}, protocol_t=1))
    assert writer.samples_path.read_bytes() == before and writer.index == 1
    assert not (writer.image_dir / "000001_uav_0.jpg").exists()
    gate = TraceSamplingGate(timeout_ns=1_000_000_000)
    base = 2_000_000_000
    inputs = dict(now_ns=base, stamps={agent: 100 for agent in AGENT_IDS},
                  previous_stamps={agent: 90 for agent in AGENT_IDS},
                  image_received_ns={agent: base - 1 for agent in AGENT_IDS},
                  image_advanced_ns={agent: base - 1 for agent in AGENT_IDS},
                  odometry_received_ns={agent: base - 1 for agent in AGENT_IDS})
    assert gate.check(**inputs) == (True, None)
    skewed = deepcopy(inputs)
    skewed["stamps"]["uav_1"] += 100_000_001
    assert gate.check(**skewed) == (False, None)
    assert gate.check(**inputs) == (True, None)
    for field, value in (("stamps", 90), ("image_received_ns", base - 300_000_000),
                         ("image_advanced_ns", base - 300_000_000),
                         ("odometry_received_ns", base - 300_000_000)):
        bad = deepcopy(inputs)
        bad[field]["uav_0"] = value
        assert gate.check(**bad) == (False, None)
        assert gate.check(**inputs) == (True, None)  # recovery resets wait
    bad = deepcopy(inputs)
    bad["stamps"]["uav_0"] = 90  # one stuck camera despite fresh callbacks
    assert gate.check(**bad) == (False, None)
    bad["now_ns"] += 1_000_000_000
    assert gate.check(**bad)[1].startswith("trace sampling timeout")
    assert gate.check(**inputs) == (True, None)
    with patch("benchmark.px4_gazebo.model_handoff.execution_trace.time.monotonic_ns", side_effect=[10, 30]), \
         patch("benchmark.px4_gazebo.model_handoff.execution_trace.time.time_ns", return_value=100):
        assert host_clock_pair() == {"monotonic_ns": 10, "monotonic_after_ns": 30, "unix_ns": 100}
    # A pause BETWEEN reads expands the bracket, without proving a clock jump.
    stream = [{"received_monotonic_ns": base, "clock_read_end_monotonic_ns": base + 1,
               "received_unix_ns": base + 100},
              {"received_monotonic_ns": base + 100, "clock_read_end_monotonic_ns": base + 500_000_100,
               "received_unix_ns": base + 500_000_150}]
    changes, widths = clock_pair_diagnostics(stream)
    assert changes == [0.] and max(widths) == 500.
    stream[1]["received_unix_ns"] += 1_000_000_000
    assert clock_pair_diagnostics(stream)[0][0] > 900.


def main():
    with tempfile.TemporaryDirectory() as directory:
        _sampling_tests(Path(directory))
        episode_dir = Path(directory) / "timing-test"
        (episode_dir / "policy").mkdir(parents=True)
        (episode_dir / "evaluator").mkdir()
        trace = ExecutionTrace(episode_dir / "policy", "timing-test", record_timing=True, record_clock_bounds=True)
        base = 1_000_000_000
        unix = 1_790_000_000_000_000_000
        for step in range(4):
            now = base + step * 50_000_000
            trace.clock_received(stamp_ns=step * 50_000_000, monotonic_ns=now, unix_ns=unix + now,
                                 monotonic_after_ns=now + 1)
            for index, agent in enumerate(AGENT_IDS):
                odometry = _odometry(now - 1_000_000)
                odometry["timestamp_us"] = (unix + now) // 1000
                odometry["timestamp_sample_us"] = odometry["timestamp_us"] - 1000
                trace.published(agent_id=agent, monotonic_ns=now + index,
                                command_timestamp_us=(unix + now) // 1000, protocol_t=step,
                                body_flu=[0., 0., 0., 0.], velocity_ned=[0., 0., 0.],
                                yaw_rate=0., odometry=odometry)
                trace.timesync_received(agent, SimpleNamespace(
                    timestamp=(unix + now)//1000, source_protocol=2, remote_timestamp=(unix + now)//1000,
                    observed_offset=-1790000000000000, estimated_offset=-1790000000000000,
                    round_trip_time=200), monotonic_ns=now + 2, unix_ns=unix + now + 2, monotonic_after_ns=now + 3)
            trace.anchor(sample_index=step, protocol_t=step, capture_monotonic_ns=now + 3,
                         image_timestamp_ns={agent: step * 50_000_000 for agent in AGENT_IDS},
                         timing={"capture_unix_ns": unix + now + 3,
                                 "clock_read_end_monotonic_ns": now + 4,
                                 "image_received_monotonic_ns": {agent: now for agent in AGENT_IDS},
                                 "odometry": {agent: odometry for agent in AGENT_IDS}})
        trace.finish(base + 200_000_000, {agent: odometry for agent in AGENT_IDS}, "land")
        trace.flush()
        header, *rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
        report = quality_report(header, rows)
        assert report_episode(episode_dir, persist=True) == report
        _reject(lambda: report_episode(episode_dir, persist=True), FileExistsError)
        assert report["receive_quality_pass"] and not report["precise_transition_supervision_ready"]
        assert report["clock_alignment"] == "unverified"
        legacy = deepcopy(header)
        legacy["schema"] = TIMED_TRACE_SCHEMA
        assert "missing_host_clock_read_bounds" in quality_report(legacy, rows)["issues"]
        h = deepcopy(header)
        for sample in h["timing_evidence"]["sim_clock"]:
            sample["clock_read_end_monotonic_ns"] += 500_000_000
        uncertain = quality_report(h, rows)
        assert "host_clock_read_uncertainty_above_draft_limit" in uncertain["issues"]
        assert "host_clock_pair_inconsistency" not in uncertain["issues"]
        h = deepcopy(header)
        h["sample_anchors"][0]["timing"]["clock_read_end_monotonic_ns"] = base - 1
        _reject(lambda: validate_trace(h, rows))
        # Incoming DDS timestamps stay untouched even with a large signed offset.
        assert rows[0]["start_odometry"]["timestamp_us"] == (unix + base)//1000
        h = deepcopy(header)
        h["sample_anchors"][0]["timing"]["image_received_monotonic_ns"]["uav_0"] = base + 4
        _reject(lambda: validate_trace(h, rows))
        h = deepcopy(header)
        h["timing_evidence"]["sim_clock"][2]["stamp_ns"] = 1
        assert "sim_clock_backward" in quality_report(h, rows)["issues"]
        h = deepcopy(header)
        h["timing_evidence"]["sim_clock"] = []
        assert "insufficient_sim_clock_evidence" in quality_report(h, rows)["issues"]
        h = deepcopy(header)
        h["sample_anchors"][2]["image_timestamp_ns"]["uav_0"] = 50_000_000
        assert "uav_0:reused_image_stamp" in quality_report(h, rows)["issues"]
        h = deepcopy(header)
        h["timing_evidence"]["sim_clock"][2]["received_unix_ns"] += 500_000_000
        assert "host_clock_pair_inconsistency" in quality_report(h, rows)["issues"]
        h = deepcopy(header)
        h["sample_anchors"][0]["timing"]["image_received_monotonic_ns"]["uav_0"] -= 300_000_000
        assert "uav_0:image_receive_age_above_draft_limit" in quality_report(h, rows)["issues"]
        # Contiguous but delayed final interval is structurally valid, quality fails.
        r = deepcopy(rows)
        for row in r[-2:]:
            row["end_monotonic_ns"] += 300_000_000
            row["duration_s"] += .3
        assert "uav_0:publication_gap_above_draft_limit" in quality_report(header, r)["issues"]
        # A bad recorder must not overwrite an existing trace.
        _reject(lambda: ExecutionTrace(Path(directory), "other", record_timing=True).flush())
    print("Trace timing quality self-test: PASS (no flight)")


if __name__ == "__main__":
    main()
