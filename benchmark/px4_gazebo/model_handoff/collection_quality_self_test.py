"""No-flight regression for opt-in quality gating and retained attempts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from .clock_probe import analyze, run_probe
from .collect import HostClockPreflightError, TraceQualityFixtureError, clock_preflight, run_fixture
from .fixtures import generate_fixtures
from .interface_self_test import _reject


def _probe_tests():
    base = 1_000_000_000
    samples = []
    for step in range(3):
        now = base + step * 50_000_000
        samples.append({"sample_monotonic_ns": now, "readings": {
            name: {"value_ns": now + offset, "before_ns": now, "after_ns": now + 100}
            for name, offset in (("unix_ns", 1_790_000_000_000_000_000), ("raw_ns", 1000), ("boottime_ns", 2000))
        }})
    assert not analyze(samples)["issues"]
    bad = deepcopy(samples)
    bad[1]["readings"]["unix_ns"]["value_ns"] += 2_400_000_000
    report = analyze(bad)
    assert len(report["events"]) == 2 and all(event["clock"] == "unix_ns" for event in report["events"])
    assert report["metrics"]["raw_ns"]["max_offset_change_outside_bounds_ms"] == 0
    uncertain = deepcopy(samples)
    uncertain[1]["readings"]["unix_ns"]["after_ns"] += 500_000_000
    assert analyze(uncertain)["issues"] == ["clock_read_uncertainty"]
    _reject(lambda: analyze(samples[:1]))


def main():
    _probe_tests()
    fixture = generate_fixtures()[0]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = SimpleNamespace(output_dir=root / "results", dataset_dir=root / "dataset", mode="collect",
                               simulator_retries=0, world="lawn", px4_root=root, auto_log_dir=root / "logs",
                               session_startup_timeout=150., shutdown_timeout=15., show_gui=False, gui_renderer="auto",
                               record_execution_trace=True, require_trace_receive_quality=True,
                               manifest=root / "manifest", runner_startup_timeout=45., perturb_probability=.15)
        final = args.output_dir / fixture.split / f"{fixture.fixture_id}.jsonl"
        attempts = args.output_dir / ".attempts" / fixture.fixture_id
        task = {"episode_id": "episode-test", "outcome": {"success": True, "termination_reason": "success"}}
        quality_ok = False
        args.clock_preflight_seconds = 20.
        with patch("benchmark.px4_gazebo.model_handoff.clock_probe.run_probe",
                   return_value={"issues": ["clock_offset_inconsistency"]}), \
             patch("benchmark.px4_gazebo.model_handoff.collect.Px4GazeboSession") as simulator:
            _reject(lambda: clock_preflight(args), HostClockPreflightError)
            assert not simulator.called
        with patch("benchmark.px4_gazebo.model_handoff.clock_probe.run_probe", return_value={"issues": []}):
            assert clock_preflight(args).parent.name == ".diagnostics"
        # Exclusive output refuses accidental evidence overwrite before polling.
        existing = root / "existing_probe.json"
        existing.write_text("saved", encoding="utf-8")
        _reject(lambda: run_probe(existing, duration_s=1), FileExistsError)
        assert existing.read_text() == "saved"

        def run(command, **kwargs):
            path = Path(command[command.index("--output") + 1])
            episode = Path(command[command.index("--record-dir") + 1]) / task["episode_id"]
            (episode / "evaluator").mkdir(parents=True)
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            return SimpleNamespace(returncode=0)

        def report(episode, *, persist):
            result = {"episode_id": task["episode_id"], "receive_quality_pass": quality_ok,
                      "issues": [] if quality_ok else ["host_clock_pair_inconsistency"]}
            if persist:
                with (episode / "evaluator" / "trace_receive_quality.json").open("x", encoding="utf-8") as out:
                    out.write(json.dumps(result))
            return result

        with patch("benchmark.px4_gazebo.model_handoff.collect.Px4GazeboSession"), \
             patch("benchmark.px4_gazebo.model_handoff.collect.subprocess.run", side_effect=run) as launched, \
             patch("benchmark.px4_gazebo.model_handoff.collect.validate_episode_directory", return_value={"samples": 28}), \
             patch("benchmark.px4_gazebo.model_handoff.trace_quality.report_episode", side_effect=report):
            _reject(lambda: run_fixture(args, fixture), TraceQualityFixtureError)
            first = attempts / "attempt_00.jsonl"
            first_bytes = first.read_bytes()
            assert not final.exists() and not args.dataset_dir.exists()
            assert (attempts / "dataset_00" / "episode-test" / "evaluator" / "trace_receive_quality.json").exists()
            assert json.loads(first.read_text())["outcome"]["success"] is True
            # Restart chooses a new attempt; rejected success is retained.
            quality_ok = True
            run_fixture(args, fixture)
            assert first.read_bytes() == first_bytes and final.is_file()
            command = launched.call_args.args[0]
            assert Path(command[command.index("--output") + 1]).name == "attempt_01.jsonl"
            assert (args.dataset_dir / "episode-test" / "evaluator" / "trace_receive_quality.json").exists()
            count = launched.call_count
            run_fixture(args, fixture)
            assert launched.call_count == count  # resume validates, no new flight
            quality_ok = False
            _reject(lambda: run_fixture(args, fixture), TraceQualityFixtureError)
            assert final.is_file() and launched.call_count == count
            # Ungated collection records warning/report but retains task success.
            args.require_trace_receive_quality = False
            args.output_dir = root / "ungated_results"
            args.dataset_dir = root / "ungated_dataset"
            run_fixture(args, fixture)
            assert (args.dataset_dir / "episode-test").is_dir()
    print("Collection quality gating + host clock probe self-test: PASS (no flight)")


if __name__ == "__main__":
    main()
