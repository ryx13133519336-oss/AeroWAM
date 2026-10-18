"""No-flight tests for opt-in traces and external/candidate interfaces."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
from types import SimpleNamespace

from .dataset import EpisodeDatasetWriter, validate_episode_directory
from .decision_interface import (Candidate, CandidateStep, CandidatePrediction, DecisionRequest, DecisionAdapter, DecisionClient, select)
from .execution_trace import ExecutionTrace, validate_trace, validate_execution_file
from .fixtures import generate_fixtures
from .interface_check import check
from .policy_client import PolicyClient, PolicyTimeoutError
from .protocol import AGENT_IDS, ModelJointAction, ModelObservation, WaypointDelta
from .self_test import _observation


def _reject(callback, exception=ValueError):
    try:
        callback()
    except exception:
        return
    raise AssertionError("Invalid input accepted")


def _odometry(received):
    return {"position": [0., 0., -2.], "velocity": [0., 0., 0.],
            "quaternion_wxyz": [1., 0., 0., 0.], "pose_frame": 1, "velocity_frame": 1,
            "timestamp_us": 10, "timestamp_sample_us": 9, "received_monotonic_ns": received}


def _publish(trace, timestamp, protocol_t):
    for index, agent in enumerate(AGENT_IDS):
        trace.published(agent_id=agent, monotonic_ns=timestamp + index,
                        command_timestamp_us=100, protocol_t=protocol_t,
                        body_flu=[.2, 0., 0., 0.], velocity_ned=[.2, 0., 0.],
                        yaw_rate=0., odometry=_odometry(timestamp - 1))


def _trace_tests(root):
    observation = _observation()
    fixture = generate_fixtures()[0].to_dict()
    writer = EpisodeDatasetWriter(root, episode_id="episode", fixture=fixture, mode="oracle", record_execution=True)
    writer.append(observation, ModelJointAction(), events=[], truth={}, protocol_t=0)
    assert not writer.frame_pair_advanced(observation.image_timestamp_ns)
    assert writer.frame_pair_advanced({agent: observation.timestamp_ns + 1 for agent in AGENT_IDS})
    before = (writer.public_dir / "samples.jsonl").read_bytes()
    _reject(lambda: writer.append(observation, ModelJointAction(), events=[], truth={}, protocol_t=1))
    assert writer.index == 1 and (writer.public_dir / "samples.jsonl").read_bytes() == before
    assert not (writer.image_dir / "000001_uav_0.jpg").exists()
    trace = ExecutionTrace(writer.public_dir, "episode")
    _publish(trace, 100, None)
    trace.anchor(sample_index=0, protocol_t=0, capture_monotonic_ns=120, image_timestamp_ns=observation.image_timestamp_ns)
    _publish(trace, 150, 0)
    trace.finish(200, {agent: _odometry(180) for agent in AGENT_IDS}, "land")
    trace.finish(200, {}, "ignored")
    trace.flush()
    trace.flush()
    summary = validate_episode_directory(root / "episode", require_execution_trace=True)
    assert summary["execution"]["agent_intervals"] == {agent: 2 for agent in AGENT_IDS}
    entries = [json.loads(line) for line in trace.path.read_text().splitlines()]
    header, rows = entries[0], entries[1:]
    assert rows[0]["protocol_t"] is None and rows[2]["protocol_t"] == 0
    assert all(row["end_monotonic_ns"] == 200 for row in rows[-2:])
    _reject(lambda: trace.anchor(sample_index=1, protocol_t=1, capture_monotonic_ns=180, image_timestamp_ns=observation.image_timestamp_ns))
    for mutate in (
        lambda h, r: h.update(finished=False),
        lambda h, r: h.update(target_pose=[0, 0, 0]),
        lambda h, r: h["clocks"].update(odometry="aligned"),
        lambda h, r: r[0].update(duration_s=1),
        lambda h, r: r[0]["start_odometry"].update(received_monotonic_ns=101),
        lambda h, r: r[0]["start_odometry"].update(pose_frame=0),
        lambda h, r: r[0]["start_odometry"].update(velocity=[float("nan"), 0, 0]),
        lambda h, r: r[2].update(start_monotonic_ns=151),
        lambda h, r: h["sample_anchors"][0].update(capture_monotonic_ns=201),
        lambda h, r: h["sample_anchors"][0]["image_timestamp_ns"].update(uav_0=-1),
    ):
        h, r = deepcopy(header), deepcopy(rows)
        mutate(h, r)
        _reject(lambda: validate_trace(h, r))
    _reject(lambda: validate_trace(header, [row for row in rows if row["agent_id"] == "uav_0"]))
    samples = [json.loads((writer.public_dir / "samples.jsonl").read_text())]
    samples[0]["protocol_t"] = 3
    _reject(lambda: validate_execution_file(trace.path, samples=samples, episode_id="episode"))
    other = ExecutionTrace(writer.public_dir, "episode")
    other.finished = True
    other.samples, other.rows = header["sample_anchors"], rows
    _reject(other.flush, FileExistsError)
    failed_dir = root / "failed"
    failed_dir.mkdir()
    failed = ExecutionTrace(failed_dir, "failed")
    _reject(failed.flush)
    failed.fail(ValueError("test failure"))
    failed.finish(10, {}, "failed")
    failed.flush()
    bad_header = json.loads(failed.path.read_text())
    assert bad_header["trace_error"] and not bad_header["finished"]
    _reject(lambda: validate_trace(bad_header, []))
    old = EpisodeDatasetWriter(root, episode_id="old", fixture=fixture, mode="oracle")
    old.append(replace(observation, episode_id="old"), ModelJointAction(), events=[], truth={})
    assert validate_episode_directory(root / "old")["samples"] == 1
    _reject(lambda: validate_episode_directory(root / "old", require_execution_trace=True))


def _decision_request():
    observation = _observation()
    candidates = tuple(Candidate(name, (CandidateStep(ModelJointAction(uav_0=WaypointDelta(dx=dx)), .5),))
                       for name, dx in (("hold", 0), ("move", .2)))
    return DecisionRequest("decision", "handoff", observation.instruction, (observation,), candidates, .5)


class _TestPredictor:
    def predict(self, *, history, instruction, candidates, horizon_s, task):
        assert instruction == history[0].instruction and horizon_s == .5 and task == "handoff"
        return [CandidatePrediction(row.candidate_id, row.steps[0].action.uav_0.dx,
                    {agent: .5 for agent in AGENT_IDS}, .4, .3, .1) for row in candidates]


def _decision_tests():
    request = _decision_request()
    assert DecisionRequest.from_wire(request.to_wire()).canonical_bytes() == request.canonical_bytes()
    adapter = DecisionAdapter(_TestPredictor())
    response = adapter(request.to_wire())
    assert check(request.to_wire(), response) == "decision"
    predictions = tuple(CandidatePrediction.from_wire(row) for row in response["predictions"])
    assert select(request, predictions) == "move"
    assert select(request, predictions, max_collision_probability=0) is None
    _reject(lambda: select(request, predictions[:1]))
    _reject(lambda: select(request, (predictions[0], predictions[0])))
    _reject(lambda: replace(predictions[0], collision_probability=float("nan")))
    _reject(lambda: replace(predictions[0], confirmation_probability=1.1))
    _reject(lambda: replace(request, horizon_s=1))
    _reject(lambda: replace(request, candidates=(request.candidates[0], request.candidates[0])))
    _reject(lambda: replace(request, instruction="different target"))
    _reject(lambda: CandidateStep(ModelJointAction(uav_0=WaypointDelta(dx=2)), .5))
    _reject(lambda: CandidateStep(ModelJointAction(uav_0=WaypointDelta(dx=1)), .1))
    raw = request.to_wire()
    raw["history"][0]["target_pose"] = [0, 0, 0]
    _reject(lambda: adapter(raw))
    raw = request.to_wire()
    raw["candidates"][0]["fixture_id"] = "oracle lookup"
    _reject(lambda: adapter(raw))
    raw = request.to_wire()
    raw["history"][0]["states"]["uav_0"]["yaw"] = float("nan")
    _reject(lambda: adapter(raw))
    raw = request.to_wire()
    raw["history"][0]["instruction"] = 5
    _reject(lambda: adapter(raw))
    assert ModelObservation.from_wire(_observation().to_wire()).canonical_bytes() == _observation().canonical_bytes()

    from .adapter_example import create_policy
    from .policy_server import ReusableThreadingTCPServer, _Handler
    with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
        server.policy = adapter
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        endpoint = f"127.0.0.1:{server.server_address[1]}"
        try:
            result, latency = DecisionClient(endpoint, timeout_s=1).predict(request)
            assert len(result) == 2 and latency < 1000
            server.policy = create_policy()
            action, _ = PolicyClient(endpoint, timeout_s=1).infer(_observation())
            assert action == ModelJointAction()
        finally:
            server.shutdown()
            worker.join(1)


def _deadline_test():
    # Each fragment arrives within the timeout; the total must still expire.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        def slow_server():
            with listener.accept()[0] as connection:
                connection.recv(4096)
                response = json.dumps(ModelJointAction().to_wire()).encode()
                try:
                    for byte in struct.pack("!I", len(response)) + response:
                        connection.sendall(bytes([byte]))
                        time.sleep(.04)
                except OSError:
                    pass
        worker = threading.Thread(target=slow_server, daemon=True)
        worker.start()
        client = PolicyClient(f"127.0.0.1:{listener.getsockname()[1]}", timeout_s=.15)
        started = time.perf_counter()
        _reject(lambda: client.infer(_observation()), PolicyTimeoutError)
        assert time.perf_counter() - started < .6
        worker.join(1)


def _flags_test():
    from .collect import _command
    from ..live_handoff_runner import validate_runtime_arguments
    from ..live_handoff_self_test import _runtime_args
    args = _runtime_args(duration=24.)
    args.record_execution_trace = True
    args.record_dir = None
    _reject(lambda: validate_runtime_arguments(args))
    args.record_dir = Path("new-records")
    validate_runtime_arguments(args)
    collect = SimpleNamespace(manifest=Path("manifest"), world="lawn", runner_startup_timeout=45,
                              perturb_probability=.15, mode="collect", record_execution_trace=True)
    command = _command(collect, generate_fixtures()[0], Path("output"), Path("records"))
    assert command.count("--record-execution-trace") == 1
    collect.record_execution_trace = False
    assert "--record-execution-trace" not in _command(collect, generate_fixtures()[0], Path("output"), Path("records"))


def main():
    with tempfile.TemporaryDirectory() as directory:
        _trace_tests(Path(directory))
    _decision_tests()
    _deadline_test()
    _flags_test()
    print("Execution trace + model interfaces self-test: PASS (no flight)")


if __name__ == "__main__":
    main()
