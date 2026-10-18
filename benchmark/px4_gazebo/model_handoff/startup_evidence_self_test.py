"""Synthetic no-flight tests for private startup evidence, not restore tests."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from .startup_evidence import StartupEvidence, validate_mode


def inputs(index=0):
    stamp = 1_000_000_000 + index * 300_000_000
    now = 10. + index * .3
    def image():
        return NS(header=NS(stamp=NS(sec=stamp // 1_000_000_000, nsec=stamp % 1_000_000_000)),
                  SerializeToString=lambda: b"synthetic-protobuf")
    return dict(odometry={i: NS(position=(0., 0., 0.), velocity=(0., 0., 0.), q=(1., 0., 0., 0.),
                                angular_velocity=(0., 0., 0.), pose_frame=1, velocity_frame=1,
                                timestamp=1000 + index, timestamp_sample=1000 + index,
                                reset_counter=0, quality=100) for i in (0, 1)},
                status={i: NS(arming_state=1, failsafe=False, nav_state=4, pre_flight_checks_pass=True) for i in (0, 1)},
                received={(kind, i): now - .01 for kind in ("status", "odometry") for i in (0, 1)},
                clock={"stamp": stamp + 20_000_000, "received": now - .01},
                cameras={f"uav_{i}": {"stamp_ns": stamp, "rgb": (image(), now - .01),
                                       "labels": (image(), now - .01)} for i in (0, 1)},
                serialize_ros=lambda msg: b"synthetic-cdr", now=now,
                host_clock={"sample_monotonic_ns": int(now * 1e9), "readings": {
                    name: {"value_ns": int(now * 1e9) + offset, "before_ns": int(now * 1e9),
                           "after_ns": int(now * 1e9) + 100} for name, offset in
                    (("unix_ns", 1_000_000_000_000), ("raw_ns", 0), ("boottime_ns", 0))}})


class StartupEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        self.capture = StartupEvidence(self.root, 2, "a" * 64)

    def test_mode_rejects_motion_and_policy_execution(self):
        base = dict(execute_flight=False, control_link_check=False, policy="rgb_scan", teammate_response="nominal")
        validate_mode(8, **base)
        for key, value in (("execute_flight", True), ("control_link_check", True), ("policy", "checkpoint"),
                           ("policy", "rgb_search"), ("teammate_response", "cautious")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_mode(8, **{**base, key: value})
        for count in (-1, 1, 21, True, 2., None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_mode(count, **base)

    def test_direct_runner_guard_precedes_ros_import(self):
        from .search_pilot import run_condition
        with self.assertRaisesRegex(ValueError, "no flight"):
            run_condition(NS(startup_evidence_frames=8, execute_flight=True, control_link_check=False,
                             policy="rgb_scan", teammate_response="nominal"), None, None)

    def test_complete_preserves_raw_hashes_and_disclaims_matching(self):
        self.capture.add(**inputs())
        self.capture.add(**inputs(1))
        result = self.capture.finish()
        self.assertTrue(result["complete"])
        for key in ("start_matched", "full_state_restored", "exposure_state_aligned", "formal_benchmark_result"):
            self.assertFalse(result[key])
        rows = [json.loads(line) for line in (self.root / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(len(row["artifacts"]), 8)
            for artifact in row["artifacts"].values():
                data = (self.root / artifact["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])
                self.assertEqual(len(data), artifact["bytes"])

    def reject(self, change):
        data = inputs()
        change(data)
        with self.assertRaises((ValueError, KeyError)):
            self.capture.add(**data)
        self.assertFalse((self.root / "frame_000").exists())
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])

    def test_stale_status(self):
        self.reject(lambda d: d["received"].update({("status", 0): d["now"] - 3}))

    def test_stale_odometry(self):
        self.reject(lambda d: d["received"].update({("odometry", 1): d["now"] - .6}))

    def test_future_receive_time(self):
        self.reject(lambda d: d["received"].update({("odometry", 0): d["now"] + 1}))

    def test_armed_or_failsafe(self):
        self.reject(lambda d: setattr(d["status"][0], "arming_state", 2))
        self.reject(lambda d: setattr(d["status"][1], "failsafe", True))

    def test_unknown_frame(self):
        self.reject(lambda d: setattr(d["odometry"][0], "velocity_frame", 2))

    def test_nonfinite_state(self):
        self.reject(lambda d: setattr(d["odometry"][1], "angular_velocity", (0., float("nan"), 0.)))

    def test_mismatched_camera_and_label(self):
        self.reject(lambda d: setattr(d["cameras"]["uav_0"]["labels"][0].header.stamp, "nsec", 1))

    def test_stale_camera(self):
        self.reject(lambda d: d["cameras"]["uav_1"].update(rgb=(d["cameras"]["uav_1"]["rgb"][0], 9.)))

    def test_clock_lag_and_staleness(self):
        self.reject(lambda d: d["clock"].update(stamp=2_000_000_000))
        self.reject(lambda d: d["clock"].update(received=9.))

    def test_reused_images_or_odometry(self):
        self.capture.add(**inputs())
        with self.assertRaisesRegex(ValueError, "stamp repeated"):
            self.capture.add(**inputs())
        data = inputs(1)
        data["odometry"][1].timestamp_sample = 1000
        with self.assertRaisesRegex(ValueError, "did not advance"):
            self.capture.add(**data)

    def test_incomplete_never_finalizes(self):
        self.capture.add(**inputs())
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.capture.finish()
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])

    def test_no_overwrite_or_extra_samples(self):
        with self.assertRaises(FileExistsError):
            StartupEvidence(self.root, 2, "a" * 64)
        self.capture.add(**inputs())
        self.capture.add(**inputs(1))
        with self.assertRaisesRegex(ValueError, "extra"):
            self.capture.add(**inputs(2))

    def test_io_failure_keeps_incomplete_artifacts(self):
        original = Path.open
        def fail(path, *args, **kwargs):
            if path.name == "uav_1_status.cdr":
                raise OSError("injected disk error")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", fail), self.assertRaises(OSError):
            self.capture.add(**inputs())
        self.assertTrue((self.root / "frame_000").exists())
        self.assertFalse(json.loads((self.root / "manifest.json").read_text())["complete"])

    def test_clock_discontinuity_does_not_complete(self):
        self.capture.add(**inputs())
        data = inputs(1)
        data["host_clock"]["readings"]["unix_ns"]["value_ns"] += 1_000_000_000
        self.capture.add(**data)
        with self.assertRaisesRegex(ValueError, "clocks inconsistent"):
            self.capture.finish()
        report = json.loads((self.root / "manifest.json").read_text())
        self.assertFalse(report["complete"])
        self.assertTrue(report["sampled_host_clock_report"]["issues"])


if __name__ == "__main__":
    unittest.main()
