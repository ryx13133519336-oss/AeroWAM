"""No-flight session clock gate regressions, including failures outside capture."""

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from .clock_probe_self_test import samples
from .startup_clock import StartupClockMonitor, finalize_startup_clock, session_quality
from .startup_evidence import StartupEvidence
from .startup_evidence_self_test import inputs

BASE = "benchmark.px4_gazebo.model_handoff.startup_clock."
ENV = {"boot_id": "same-boot", "clocksource": "tsc", "uptime": "1 2"}


class SessionQualityTests(unittest.TestCase):
    def test_stable_interval_passes_without_mutating_samples(self):
        rows = samples([0, 0, 0])
        original = deepcopy(rows)
        self.assertTrue(session_quality(rows, ENV, ENV)["quality_passed"])
        self.assertEqual(original, rows)

    def test_drift_before_or_after_capture_is_not_hidden(self):
        for rows in (samples([20_000_000, 0, 0, 0]), samples([0, 0, 0, 20_000_000])):
            self.assertTrue(session_quality(rows[1:3], ENV, ENV)["quality_passed"])
            self.assertFalse(session_quality(rows, ENV, ENV)["quality_passed"])

    def test_accumulated_drift_and_roundtrip_fail(self):
        for offsets in ([i*1_000_000 for i in range(21)],
                        [i*1_000_000 for i in range(21)] + [0]):
            quality = session_quality(samples(offsets), ENV, ENV)
            self.assertFalse(quality["quality_passed"])
            self.assertIn("clock_offset_excursion", quality["issues"])

    def test_poll_gap_exact_boundary_and_one_ns_excess(self):
        for gap, expected in ((250_000_000, True), (250_000_001, False)):
            quality = session_quality(samples([0, 0], interval_ns=gap), ENV, ENV)
            self.assertEqual(quality["quality_passed"], expected)

    def test_boot_or_clocksource_change_or_absence_fails(self):
        for after in ({**ENV, "boot_id": "different"}, {**ENV, "clocksource": "other"},
                      {**ENV, "boot_id": None}, {**ENV, "clocksource": None}):
            self.assertIn("clock_environment_unverified", session_quality(samples([0, 0]), ENV, after)["issues"])

    def test_missing_required_clock_fails(self):
        rows = samples([0, 0])
        del rows[1]["readings"]["raw_ns"]
        self.assertIn("required_host_clock_unavailable", session_quality(rows, ENV, ENV)["issues"])

    def test_insufficient_malformed_or_nonincreasing_samples_fail(self):
        malformed = samples([0, 0])
        malformed[1]["sample_monotonic_ns"] = malformed[0]["sample_monotonic_ns"]
        for rows in ([], samples([0]), malformed, [{}]):
            quality = session_quality(rows, ENV, ENV)
            self.assertFalse(quality["quality_passed"])

    def test_worker_error_is_not_stable_data(self):
        quality = session_quality(samples([0, 0]), ENV, ENV, ["injected error"])
        self.assertFalse(quality["quality_passed"])
        self.assertIn("clock_monitor_error", quality["issues"])


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "clock.json"
        self.env = patch(BASE + "environment", return_value=ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_endpoint_samples_and_exclusive_output_with_digest(self):
        # No wall-clock sleeps: block the fake worker until finish requests stop.
        monitor = StartupClockMonitor(self.path)
        with patch(BASE + "sample_clocks", side_effect=samples([0, 0])), \
             patch.object(monitor, "_worker", side_effect=lambda: monitor._stop.wait()):
            monitor.start()
            quality = monitor.finish()
        saved = json.loads(self.path.read_bytes())
        self.assertTrue(quality["quality_passed"])
        self.assertEqual(len(saved["samples"]), 2)
        self.assertEqual(saved["report"], quality["clock_report"])
        self.assertEqual(quality["artifact"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertIs(monitor.finish(), quality)
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            StartupClockMonitor(self.path).start()
        self.assertEqual(self.path.read_bytes(), original)

    def test_worker_sampling_exception_preserves_partial_data_and_fails(self):
        monitor = StartupClockMonitor(self.path)
        entered = threading.Event()
        def read():
            if threading.current_thread() is threading.main_thread():
                return samples([0, 0])[int(monitor._stop.is_set())]
            entered.set()
            raise OSError("read failure")
        with patch(BASE + "sample_clocks", side_effect=read), patch(BASE + "INTERVAL_S", .001):
            monitor.start()
            self.assertTrue(entered.wait(2.))
            quality = monitor.finish()
        self.assertFalse(quality["quality_passed"])
        self.assertTrue(json.loads(self.path.read_bytes())["interrupted"])
        self.assertEqual(len(json.loads(self.path.read_bytes())["samples"]), 2)
        with self.assertRaises(RuntimeError):
            monitor.check()

    def test_capacity_limit_fails_instead_of_dropping_history(self):
        monitor = StartupClockMonitor(self.path)
        with patch(BASE + "MAX_SAMPLES", 1), patch(BASE + "sample_clocks", side_effect=samples([0, 0])), \
             patch.object(monitor, "_worker", side_effect=lambda: monitor._stop.wait()):
            monitor.start()
            quality = monitor.finish()
        self.assertFalse(quality["quality_passed"])
        self.assertEqual(len(json.loads(self.path.read_bytes())["samples"]), 1)

    def test_initial_read_failure_can_never_finalize_as_passed(self):
        monitor = StartupClockMonitor(self.path)
        with patch(BASE + "sample_clocks", side_effect=[OSError("initial"), samples([0])[0]]):
            with self.assertRaises(OSError):
                monitor.start()
            quality = monitor.finish()
        self.assertFalse(quality["quality_passed"])
        self.assertTrue(quality["monitor_errors"])

    def test_unstoppable_worker_never_marks_quality_passed(self):
        monitor = StartupClockMonitor(self.path)
        with patch(BASE + "sample_clocks", side_effect=samples([0, 0])), \
             patch.object(monitor, "_worker", side_effect=lambda: monitor._stop.wait()):
            monitor.start()
            real_thread = monitor._thread
            monitor._stop.set()
            real_thread.join(timeout=2.)
            monitor._thread = NS(ident=1, join=lambda **kwargs: None, is_alive=lambda: True)
            quality = monitor.finish()
        self.assertFalse(quality["quality_passed"])
        self.assertIn("Clock monitor thread did not stop", quality["monitor_errors"])

    def test_persist_failure_closes_handle(self):
        monitor = StartupClockMonitor(self.path)
        with patch(BASE + "sample_clocks", side_effect=samples([0, 0])), \
             patch.object(monitor, "_worker", side_effect=lambda: monitor._stop.wait()):
            monitor.start()
            with patch.object(monitor._stream, "write", side_effect=OSError("disk")), self.assertRaises(OSError):
                monitor.finish()
        self.assertTrue(monitor._stream.closed)


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        self.capture = StartupEvidence(self.root, 2, "a" * 64, require_session_clock=True)
        self.summary = {"category": "startup_only", "reason": "no_motion_commands_requested",
                        "owned_processes_stopped": True, "cleanup_errors": []}

    def captured(self):
        for i in range(2):
            self.capture.add(**inputs(i))
        result = self.capture.finish()
        self.assertTrue(result["capture_complete"])
        self.assertFalse(result["complete"])
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])

    def finish(self, quality, *, evidence=True):
        finalize_startup_clock(NS(finish=lambda: quality), self.summary, self.capture if evidence else None)

    def test_only_clean_session_and_full_interval_can_complete(self):
        self.captured()
        self.finish(session_quality(samples([0, 0]), ENV, ENV))
        self.assertTrue(self.summary["startup_receive_quality_passed"])
        self.assertTrue(json.loads((self.root / "manifest.json").read_text())["complete"])
        self.assertEqual(self.summary["category"], "startup_only")
        for key in ("start_matched", "full_state_restored", "exposure_state_aligned"):
            self.assertFalse(self.summary["startup_evidence"][key])

    def test_capture_pass_but_cleanup_drift_rejects_and_keeps_raw_files(self):
        self.captured()
        self.finish(session_quality(samples([0, 0, 20_000_000]), ENV, ENV))
        self.assertEqual(self.summary["category"], "simulator_error")
        self.assertFalse(self.summary["startup_receive_quality_passed"])
        saved = json.loads((self.root / "manifest.json").read_text())
        self.assertTrue(saved["capture_complete"])
        self.assertFalse(saved["complete"])
        self.assertTrue((self.root / "frame_001/uav_1_status.cdr").exists())

    def test_cleanup_error_or_running_owned_process_blocks_completion(self):
        self.captured()
        for changes in ({"cleanup_errors": ["shutdown failed"]}, {"owned_processes_stopped": False}):
            self.summary.update(category="startup_only", reason="no_motion_commands_requested",
                                owned_processes_stopped=True, cleanup_errors=[])
            self.summary.update(changes)
            self.finish(session_quality(samples([0, 0]), ENV, ENV))
            self.assertFalse(self.summary["startup_receive_quality_passed"])

    def test_startup_failure_reason_preserved_even_if_clock_also_fails(self):
        self.summary.update(category="simulator_error", reason="streams_not_ready")
        self.finish(session_quality(samples([0, 20_000_000]), ENV, ENV), evidence=False)
        self.assertEqual(self.summary["reason"], "streams_not_ready")
        self.assertFalse(self.summary["startup_receive_quality_passed"])

    def test_absent_or_incomplete_capture_never_passes_with_good_clocks(self):
        for evidence in (True, False):
            self.finish(session_quality(samples([0, 0]), ENV, ENV), evidence=evidence)
            self.assertFalse(self.summary["startup_receive_quality_passed"])

    def test_monitor_write_failure_cannot_leave_completed_manifest(self):
        self.captured()
        def failed():
            raise OSError("cannot save probe")
        finalize_startup_clock(NS(finish=failed), self.summary, self.capture)
        self.assertEqual(self.summary["category"], "simulator_error")
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])

    def test_manifest_write_failure_cannot_leave_provisional_success(self):
        self.captured()
        with patch.object(self.capture, "_write_manifest", side_effect=OSError("disk full")):
            self.finish(session_quality(samples([0, 0]), ENV, ENV))
        self.assertFalse(self.summary["startup_receive_quality_passed"])
        self.assertEqual(self.summary["category"], "simulator_error")
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])


class RunnerLifecycleTests(unittest.TestCase):
    """Run the real runner with in-memory transports/processes; no ROS installed."""

    def run_fake(self, *, drift=False, initial_error=False, cleanup_error=False, process_cleanup_error=False):
        from . import search_pilot as pilot
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            executable = root / "build/px4_sitl_default/bin/px4"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"not an executable; Popen is mocked")
            events, callbacks, created = [], [], []
            index = [0]
            class Image:
                def CopyFrom(self, other):
                    self.__dict__.update(other.__dict__)
                def SerializeToString(self):
                    return b"fake-image"
            def deliver(topic, callback):
                data = inputs(index[0])
                if "vehicle_command_ack" in topic or "/contacts/" in topic:
                    return
                i = int("smoke_1" in topic)
                if "vehicle_status" in topic:
                    callback(data["status"][i])
                elif "vehicle_odometry" in topic:
                    callback(data["odometry"][i])
                else:
                    stamp = next(iter(data["cameras"].values()))["stamp_ns"]
                    t = NS(sec=stamp // 10**9, nsec=stamp % 10**9)
                    msg = NS(sim=t) if topic.endswith("/clock") else NS(
                        header=NS(stamp=t), width=256, height=256, step=768, data=b"fake")
                    callback(msg)
            def subscribe(msg_type, topic, callback, *args):
                callbacks.append((topic, callback))
                deliver(topic, callback)
                deliver(topic, callback)
                return True
            def advance(_):
                index[0] += 1
                for topic, callback in callbacks:
                    deliver(topic, callback)
            ros = NS(create_subscription=subscribe, destroy_node=Mock(
                side_effect=OSError("ROS shutdown failure") if cleanup_error else None))
            node = NS(subscribe=subscribe, unsubscribe=lambda topic: None)
            executor = NS(add_node=lambda node: None, shutdown=lambda **kwargs: None)
            class Monitor:
                def __init__(self, path):
                    pass
                def start(self):
                    events.append("clock_started")
                def check(self):
                    pass
                def finish(self):
                    assert events[-1] == "owned_processes_stopped"
                    events.append("clock_finished")
                    return session_quality(samples([0, 0, 20_000_000 if drift else 0]), ENV, ENV)
            def popen(*args, **kwargs):
                assert "clock_started" in events
                events.append("process_started")
                process = NS(returncode=None)
                process.poll = lambda: process.returncode
                created.append(process)
                return process
            def stop(processes):
                for _, process in processes:
                    process.returncode = 0
                if process_cleanup_error and processes and processes[0][0] != "gui":
                    events.append("process_cleanup_failed")
                    raise OSError("injected process cleanup failure")
                events.append("owned_processes_stopped")
                return []
            modules = {
                "rclpy": NS(init=lambda **kwargs: None, create_node=lambda _: ros, shutdown=lambda: None),
                "rclpy.executors": NS(SingleThreadedExecutor=lambda: executor),
                "rclpy.qos": NS(DurabilityPolicy=NS(VOLATILE=0), ReliabilityPolicy=NS(BEST_EFFORT=0),
                                QoSProfile=lambda **kwargs: None),
                "rclpy.serialization": NS(serialize_message=lambda _: b"fake-cdr"),
                "px4_msgs": NS(__path__=[]), "px4_msgs.msg": NS(VehicleOdometry=object,
                    VehicleStatus=object, VehicleCommand=object, VehicleCommandAck=object),
                "gz": NS(__path__=[]), "gz.transport13": NS(Node=lambda: node),
                "gz.msgs10": NS(__path__=[]), "gz.msgs10.image_pb2": NS(Image=Image),
                "gz.msgs10.clock_pb2": NS(Clock=object), "gz.msgs10.contacts_pb2": NS(Contacts=object)}
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch("benchmark.px4_gazebo.live_sim_manager.read_process_table", return_value=[]))
            import xml.etree.ElementTree as ET
            stack.enter_context(patch.object(pilot, "search_world", return_value=ET.ElementTree(ET.Element("sdf"))))
            stack.enter_context(patch.object(pilot, "verify", return_value={}))
            stack.enter_context(patch.object(pilot, "scene_digest", return_value="a"*64))
            stack.enter_context(patch.object(pilot, "StartupClockMonitor", Monitor))
            stack.enter_context(patch.object(pilot, "stop_owned", side_effect=stop))
            stack.enter_context(patch.object(pilot.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(pilot.threading, "Thread", return_value=NS(
                start=lambda: None, join=lambda **kwargs: None)))
            stack.enter_context(patch.object(pilot.time, "sleep", side_effect=advance))
            stack.enter_context(patch("benchmark.px4_gazebo.model_handoff.clock_probe.sample_clocks",
                                      side_effect=samples([0, 0])))
            stack.enter_context(patch.object(pilot, "RGBScanPolicy", side_effect=OSError("setup failed")
                                             if initial_error else None))
            stack.enter_context(patch.object(pilot, "pair_rgb_instance", return_value=None))
            stack.enter_context(patch.object(pilot, "resize_pair", return_value=NS(object_labels=None)))
            stack.enter_context(patch.object(pilot, "annotate", return_value=NS(frame=None)))
            stack.enter_context(patch.object(pilot, "ros_image_to_jpeg", return_value=b"fake-jpeg"))
            stack.enter_context(redirect_stdout(StringIO()))
            args = NS(startup_evidence_frames=2, execute_flight=False, control_link_check=False,
                      policy="rgb_scan", teammate_response="nominal", px4_root=root,
                      models=root, agent="separate-fake-agent", duration=30., agent_runtime={},
                      dds_transport="udp_all", ros_domain_id=78, gui=process_cleanup_error, startup_timeout=10.)
            result = pilot.run_condition(args, pilot.manual_condition("Find the blue cube", "blue"), root / "run")
            saved = json.loads((root / "run/evaluator/summary.json").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(events[-1], "clock_finished")
            self.assertTrue(all(p.poll() is not None for p in created))
            if not initial_error:
                manifest = json.loads((root / "run/evaluator/startup_evidence/manifest.json").read_text())
                self.assertEqual(manifest["complete"], not drift and not cleanup_error and not process_cleanup_error)
            return result, events

    def test_success_is_finalized_only_after_owned_process_cleanup(self):
        result, events = self.run_fake()
        self.assertTrue(result["startup_receive_quality_passed"])
        self.assertEqual(events[0], "clock_started")
        self.assertEqual(result["category"], "startup_only")

    def test_cleanup_period_clock_drift_rejects_actual_runner(self):
        result, _ = self.run_fake(drift=True)
        self.assertEqual(result["category"], "simulator_error")
        self.assertFalse(result["startup_receive_quality_passed"])

    def test_error_before_spawn_still_finishes_monitor(self):
        result, events = self.run_fake(initial_error=True)
        self.assertNotIn("process_started", events)
        self.assertEqual(result["category"], "simulator_error")
        self.assertIn("setup failed", result["reason"])
        self.assertFalse(result["startup_receive_quality_passed"])

    def test_ros_cleanup_error_never_marks_capture_complete(self):
        result, _ = self.run_fake(cleanup_error=True)
        self.assertEqual(result["category"], "simulator_error")
        self.assertTrue(result["cleanup_errors"])
        self.assertFalse(result["startup_receive_quality_passed"])

    def test_process_cleanup_error_still_cleans_owned_gui_and_rejects(self):
        result, events = self.run_fake(process_cleanup_error=True)
        self.assertIn("process_cleanup_failed", events)
        self.assertTrue(result["owned_processes_stopped"])
        self.assertTrue(result["cleanup_errors"])
        self.assertEqual(result["category"], "simulator_error")
        self.assertFalse(result["startup_receive_quality_passed"])


if __name__ == "__main__":
    unittest.main()
