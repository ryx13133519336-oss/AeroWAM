"""Registered service regression using synthetic Python backends, no models."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import py_compile
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from unittest.mock import patch

from .adapter_manifest_self_test import synthetic_manifest, prediction_io
from .adapter_runtime import RegisteredRuntime, RegisteredHandler
from .decision_interface import DecisionClient
from .grounded_interface import GroundedAction
from .language_tasks import TargetReport
from .policy_client import PolicyClient, PolicyTimeoutError, _receive_exact
from .policy_server import ReusableThreadingTCPServer
from .protocol import ModelJointAction
from .self_test import _observation


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = synthetic_manifest(self.root)
        self.manifest["implementation"] = "transport_example"
        self.name = "synthetic_runtime_" + uuid.uuid4().hex
        self.manifest["factory"] = self.name + ":create"
        self.path = self.root / "manifest.json"
        self.response = ModelJointAction().to_wire()
        self.set_source()

    def set_source(self, source=None):
        if source is None:
            source = ("CALLS = 0\ndef create():\n    def policy(payload):\n"
                      "        global CALLS\n        CALLS += 1\n        return " + repr(self.response) + "\n    return policy\n")
        (self.root / "adapter.py").write_text(source, encoding="utf-8")
        self.manifest["artifacts"][0]["sha256"] = hashlib.sha256((self.root / "adapter.py").read_bytes()).hexdigest()

    def save(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def runtime(self):
        self.save()
        runtime = RegisteredRuntime(self.path, "adapter_code", self.root / "run")
        self.addCleanup(runtime.close)
        return runtime

    def events(self):
        return [json.loads(s) for s in (self.root / "run/events.jsonl").read_text().splitlines()]

    def server(self, runtime):
        server = ReusableThreadingTCPServer(("127.0.0.1", 0), RegisteredHandler)
        server.policy = runtime
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(stop)
        return f"127.0.0.1:{server.server_address[1]}"

    def test_compiles_verified_snapshot_and_records_process_without_model_claim(self):
        runtime = self.runtime()
        report = json.loads((self.root / "run/startup.json").read_bytes())
        self.assertEqual(report["pid"], os.getpid())
        self.assertEqual(report["executable"], sys.executable)
        self.assertTrue(report["entry_source_compiled"])
        self.assertTrue(report["listed_artifacts_verified_after_factory"])
        self.assertEqual(hashlib.sha256((self.root / "run/entry_source.py").read_bytes()).hexdigest(), report["compiled_entry_sha256"])
        for field in ("weight_consumption_verified", "dependency_completeness_verified", "gpu_execution_verified",
                      "backend_cancellation_supported", "end_to_end_deadline_verified", "role_free_execution_verified",
                      "prediction_quality_verified", "flight_ready", "formal_benchmark_result"):
            self.assertIs(report[field], False)
        self.assertEqual(runtime(_observation().to_wire()), self.response)

    def test_stale_bytecode_is_not_executed(self):
        self.response["waypoints"]["uav_0"][0] = .1
        self.set_source()
        entry = self.root / "adapter.py"
        stamp = entry.stat()
        py_compile.compile(str(entry), doraise=True)
        self.response["waypoints"]["uav_0"][0] = .2
        self.set_source()
        self.assertEqual(entry.stat().st_size, stamp.st_size)
        os.utime(entry, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        runtime = self.runtime()
        self.assertEqual(runtime(_observation().to_wire())["waypoints"]["uav_0"][0], .2)

    def test_wrong_hash_prevents_entry_execution_and_preserves_failure(self):
        self.save()
        (self.root / "adapter.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            RegisteredRuntime(self.path, "adapter_code", self.root / "run")
        self.assertNotIn(self.name, sys.modules)
        self.assertIn("startup_failed", [r["event"] for r in self.events()])
        self.assertFalse((self.root / "run/startup.json").exists())

    def test_factory_must_be_defined_in_selected_source(self):
        self.set_source("from functools import partial as create\n")
        with self.assertRaisesRegex(ValueError, "defined in the selected"):
            self.runtime()
        self.assertNotIn(self.name, sys.modules)

    def test_preimported_entry_is_not_silently_reused(self):
        prior = types.ModuleType(self.name)
        sys.modules[self.name] = prior
        self.addCleanup(sys.modules.pop, self.name, None)
        with self.assertRaisesRegex(ValueError, "already imported"):
            self.runtime()
        self.assertIs(sys.modules[self.name], prior)

    def test_factory_error_is_not_ready(self):
        self.set_source("def create():\n    raise RuntimeError('synthetic init failure')\n")
        with self.assertRaisesRegex(RuntimeError, "synthetic init"):
            self.runtime()
        self.assertEqual(self.events()[-1]["event"], "service_closed")
        self.assertTrue(self.events()[-1]["faulted"])

    def test_noncallable_factory_result_is_not_ready(self):
        self.set_source("def create():\n    return 7\n")
        with self.assertRaisesRegex(TypeError, "return a callable"):
            self.runtime()
        self.assertFalse((self.root / "run/startup.json").exists())

    def test_changed_config_during_initialization_refuses_service(self):
        config = str(self.root / "config.json")
        self.set_source("def create():\n    from pathlib import Path\n    Path(" + repr(config) +
                        ").write_text('changed')\n    return lambda payload: {}\n")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.runtime()
        self.assertFalse((self.root / "run/startup.json").exists())

    def test_unavailable_private_input_never_reaches_backend(self):
        runtime = self.runtime()
        request = _observation().to_wire()
        request["target_pose"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            runtime(request)
        self.assertEqual(runtime._module.CALLS, 0)
        self.assertIsNone(runtime.episode_id)
        self.assertFalse(runtime.faulted)

    def test_cross_episode_reuse_is_refused_before_backend(self):
        runtime = self.runtime()
        request = _observation().to_wire()
        runtime(request)
        request["episode_id"] = "another_episode"
        with self.assertRaisesRegex(ValueError, "Cross-episode"):
            runtime(request)
        self.assertEqual(runtime._module.CALLS, 1)
        self.assertEqual(runtime.episode_id, "episode")

    def test_adapter_mutation_cannot_forge_report_reference(self):
        obs = _observation()
        self.manifest["interface"] = "grounded_action"
        self.response = GroundedAction(ModelJointAction(), (TargetReport("uav_0", 7, (.1, .1, .3, .3), .8, "found"),)).to_wire()
        self.set_source("def create():\n    def policy(payload):\n"
                        "        payload['image_timestamp_ns']['uav_0'] = 7\n        return " + repr(self.response) + "\n    return policy\n")
        runtime = self.runtime()
        request = obs.to_wire()
        with self.assertRaisesRegex(ValueError, "observed frame"):
            runtime(request)
        self.assertEqual(request["image_timestamp_ns"], obs.image_timestamp_ns)
        self.assertTrue(runtime.faulted)

    def test_invalid_response_faults_service_no_fallback(self):
        self.response = {"unexpected": "not an action"}
        self.set_source()
        runtime = self.runtime()
        with self.assertRaises(ValueError):
            runtime(_observation().to_wire())
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            runtime(_observation().to_wire())
        self.assertEqual(runtime._module.CALLS, 1)

    def test_busy_request_is_rejected_not_queued(self):
        runtime = self.runtime()
        runtime._call_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                runtime(_observation().to_wire())
            self.assertEqual(runtime._module.CALLS, 0)
        finally:
            runtime._call_lock.release()

    def test_expired_deadline_does_not_invoke_backend(self):
        runtime = self.runtime()
        with self.assertRaises(TimeoutError):
            runtime(_observation().to_wire(), deadline=time.perf_counter() - 1)
        self.assertEqual(runtime._module.CALLS, 0)

    def test_audit_io_failure_prevents_execution_and_faults_service(self):
        runtime = self.runtime()
        with patch.object(runtime, "event", side_effect=OSError("synthetic audit IO failure")):
            with self.assertRaises(OSError):
                runtime(_observation().to_wire())
        self.assertEqual(runtime._module.CALLS, 0)
        self.assertTrue(runtime.faulted)

    def test_actual_tcp_joint_action_and_input_rejection(self):
        runtime = self.runtime()
        endpoint = self.server(runtime)
        client = PolicyClient(endpoint)
        self.assertEqual(client.timeout_s, .4)
        self.assertEqual(client.infer(_observation())[0], ModelJointAction())
        request = _observation().to_wire()
        request["global_map"] = []
        with self.assertRaises(RuntimeError):
            client.request(json.dumps(request).encode())
        self.assertEqual(runtime._module.CALLS, 1)
        self.assertFalse(runtime.faulted)

    def test_actual_tcp_candidate_protocol(self):
        self.manifest["interface"] = "candidate_prediction"
        self.manifest["required_inputs"] += ["observed_history", "candidate_actions", "horizon_s", "task_kind"]
        request, self.response = prediction_io()
        self.set_source()
        runtime = self.runtime()
        from .decision_interface import DecisionRequest
        client = DecisionClient(self.server(runtime))
        predictions, latency = client.predict(DecisionRequest.from_wire(request))
        self.assertEqual([p.candidate_id for p in predictions], ["a", "b"])
        self.assertLess(latency, 400)

    def test_actual_tcp_grounded_protocol(self):
        self.manifest["interface"] = "grounded_action"
        self.response = GroundedAction(ModelJointAction()).to_wire()
        self.set_source()
        runtime = self.runtime()
        result, latency = PolicyClient(self.server(runtime)).infer_grounded(_observation())
        self.assertEqual(result.reports, ())
        self.assertLess(latency, 400)

    def test_entire_tcp_exchange_is_serial_and_busy_does_not_fault_backend(self):
        runtime = self.runtime()
        client = PolicyClient(self.server(runtime))
        runtime.exchange_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                client.infer(_observation())
            self.assertEqual(runtime._module.CALLS, 0)
            self.assertFalse(runtime.faulted)
        finally:
            runtime.exchange_lock.release()
        self.assertEqual(client.infer(_observation())[0], ModelJointAction())
        self.assertEqual(runtime._module.CALLS, 1)

    def test_close_during_backend_does_not_claim_cancellation_or_return_response(self):
        self.set_source("import threading\nENTERED = threading.Event()\nRELEASE = threading.Event()\n"
                        "def create():\n    def policy(payload):\n        ENTERED.set()\n        RELEASE.wait(5)\n"
                        "        return " + repr(self.response) + "\n    return policy\n")
        runtime = self.runtime()
        module = runtime._module
        request, results, errors = _observation().to_wire(), [], []
        def call():
            try:
                results.append(runtime(request))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        try:
            self.assertTrue(module.ENTERED.wait(2))
            runtime.close()
            self.assertTrue(self.events()[-1]["backend_active_at_close"])
            self.assertIs(self.events()[-1]["backend_cancelled"], False)
        finally:
            module.RELEASE.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(results)
        self.assertIsInstance(errors[0], RuntimeError)

    def test_actual_tcp_slow_backend_faults_and_never_reuses_state(self):
        self.set_source("CALLS = 0\ndef create():\n    def policy(payload):\n        global CALLS\n        CALLS += 1\n"
                        "        import time\n        time.sleep(.5)\n        return " + repr(self.response) + "\n    return policy\n")
        runtime = self.runtime()
        client = PolicyClient(self.server(runtime))
        with self.assertRaises((PolicyTimeoutError, ConnectionError)):
            client.infer(_observation())
        limit = time.perf_counter() + 2
        while runtime.backend_active and time.perf_counter() < limit:
            time.sleep(.01)
        self.assertFalse(runtime.backend_active)
        self.assertTrue(runtime.faulted)
        with self.assertRaises(RuntimeError):
            client.infer(_observation())
        self.assertEqual(runtime._module.CALLS, 1)

    def test_stalled_packet_and_duplicate_json_never_invoke_backend(self):
        runtime = self.runtime()
        endpoint = self.server(runtime)
        host, port = endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=1) as connection:
            connection.sendall(struct.pack("!I", 100) + b"{")
            first = connection.recv(1)
            # OS socket timeout rounding can leave enough deadline to send an
            # explicit error; EOF and a framed timeout are both valid refusal.
            if first:
                size = struct.unpack("!I", first + _receive_exact(connection, 3))[0]
                error = json.loads(_receive_exact(connection, size))
                self.assertEqual(error["error"], "TimeoutError")
        with self.assertRaises(RuntimeError):
            PolicyClient(endpoint).request(b'{"protocol":"a","protocol":"b"}')
        self.assertEqual(runtime._module.CALLS, 0)
        failures = [r for r in self.events() if r["event"] == "transport_failed"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(r["entered_runtime"] is False for r in failures))

    def test_response_send_failure_faults_backend_before_reuse(self):
        runtime = self.runtime()
        raw = json.dumps(_observation().to_wire()).encode()
        class BrokenSendSocket:
            def __init__(self):
                self.data = struct.pack("!I", len(raw)) + raw
            def settimeout(self, value):
                pass
            def recv(self, size):
                result, self.data = self.data[:size], self.data[size:]
                return result
            def sendall(self, value):
                raise OSError("synthetic client disconnect")
        RegisteredHandler(BrokenSendSocket(), ("127.0.0.1", 1), types.SimpleNamespace(policy=runtime))
        self.assertTrue(runtime.faulted)
        self.assertEqual(runtime._module.CALLS, 1)
        with self.assertRaises(RuntimeError):
            runtime(_observation().to_wire())

    def test_existing_run_dir_is_never_overwritten(self):
        runtime = self.runtime()
        before = (self.root / "run/startup.json").read_bytes()
        with self.assertRaises(FileExistsError):
            RegisteredRuntime(self.path, "adapter_code", self.root / "run")
        self.assertEqual((self.root / "run/startup.json").read_bytes(), before)
        runtime.close()
        runtime.close()
        self.assertNotIn(self.name, sys.modules)

    def test_manifest_cli_starts_separate_process_and_serves(self):
        self.save()
        command = [sys.executable, "-B", "-u", "-m", "benchmark.px4_gazebo.model_handoff.adapter_server",
                   "--manifest", str(self.path), "--entry-artifact", "adapter_code",
                   "--run-dir", str(self.root / "run"), "--port", "0"]
        stdout, stderr = self.root / "stdout.log", self.root / "stderr.log"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err)
            try:
                endpoint = None
                limit = time.perf_counter() + 15
                while time.perf_counter() < limit and process.poll() is None:
                    if (self.root / "run/events.jsonl").exists():
                        for row in self.events():
                            if row["event"] == "listening":
                                endpoint = f"{row['address'][0]}:{row['address'][1]}"
                    if endpoint:
                        break
                    time.sleep(.02)
                self.assertIsNotNone(endpoint, stderr.read_text(errors="replace"))
                self.assertEqual(PolicyClient(endpoint).infer(_observation())[0], ModelJointAction())
                startup = json.loads((self.root / "run/startup.json").read_bytes())
                self.assertEqual(startup["pid"], process.pid)
                self.assertNotEqual(startup["pid"], os.getpid())
                self.assertFalse(startup["flight_ready"])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.assertIsNotNone(process.returncode)


if __name__ == "__main__":
    unittest.main()
