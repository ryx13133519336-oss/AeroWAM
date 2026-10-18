"""Synthetic offline diagnostics only; no simulator/model or system clock query."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .clock_probe import PROBE_SCHEMA
from .clock_probe_self_test import samples
from .start_comparison import (PURPOSE, SPEC_SCHEMA, canonical, compare_startup, compare_states,
                               inspect_startup, validate_spec)
from .startup_clock import session_quality
from .startup_evidence import StartupEvidence
from .startup_evidence_self_test import inputs


def spec():
    return {"schema": SPEC_SCHEMA, "purpose": PURPOSE, "tolerances": {
        "position_l2_m": .1, "velocity_l2_m_s": .1, "attitude_geodesic_rad": .05,
        "angular_velocity_l2_rad_s": .1, "quaternion_norm_error": .0001}}


ENV = {"boot_id": "synthetic-boot", "clocksource": "synthetic-tsc", "uptime": "0 0"}


def capture(root, *, version=2, drift=False):
    """Explicit fake serialized bytes; no protobuf/CDR validity is claimed."""
    with patch("benchmark.px4_gazebo.model_handoff.startup_evidence.environment", return_value=ENV):
        evidence = StartupEvidence(root, 2, "a" * 64, require_session_clock=version == 2)
        evidence.add(**inputs())
        evidence.add(**inputs(1))
        evidence.finish()
    if version == 2:
        clocks = samples([0] * 8)
        # Capture input fixtures use monotonic 10.0 and 10.3 seconds.
        for row in clocks:
            row["sample_monotonic_ns"] += 9_000_000_000
            for reading in row["readings"].values():
                for key in ("value_ns", "before_ns", "after_ns"):
                    reading[key] += 9_000_000_000
        if drift:
            clocks[-1]["readings"]["raw_ns"]["value_ns"] += 20_000_000
        quality = session_quality(clocks, ENV, ENV)
        raw = canonical({"schema": PROBE_SCHEMA, "environment": ENV, "environment_after": ENV,
                         "interrupted": False, "samples": clocks, "report": quality["clock_report"],
                         "session_quality": quality})
        (root.parent / "startup_host_clocks.json").write_bytes(raw)
        evidence.finish_session({**quality, "artifact": {"path": "startup_host_clocks.json",
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}, session_succeeded=True)
    return root


class StateTests(unittest.TestCase):
    def setUp(self):
        data = inputs()
        self.state = {f"uav_{i}": {
            "position_ned_m": list(data["odometry"][i].position),
            "velocity_ned_m_s": list(data["odometry"][i].velocity),
            "quaternion_wxyz": list(data["odometry"][i].q),
            "angular_velocity_rad_s": list(data["odometry"][i].angular_velocity),
            "pose_frame": 1, "velocity_frame": 1, "timestamp_us": 1000, "timestamp_sample_us": 1000,
            "reset_counter": 0, "quality": 100, "arming_state": 1, "nav_state": 4,
            "failsafe": False, "pre_flight_checks_pass": True} for i in (0, 1)}
        self.other = deepcopy(self.state)

    def compare(self):
        return compare_states(self.state, self.other, spec())["uav_0"]

    def test_identical_numerical_state(self):
        self.assertTrue(self.compare()["all_numeric_checks_passed"])
        self.assertTrue(self.compare()["all_discrete_checks_passed"])

    def test_l2_not_componentwise(self):
        self.other["uav_0"]["position_ned_m"] = [.08, .08, 0]
        result = self.compare()
        self.assertAlmostEqual(result["raw_coordinate_differences"]["position_l2_m"], math.hypot(.08, .08))
        self.assertFalse(result["within_tolerances"]["position_l2_m"])

    def test_boundary_is_inclusive(self):
        self.other["uav_0"]["velocity_ned_m_s"][0] = .1
        self.assertTrue(self.compare()["within_tolerances"]["velocity_l2_m_s"])
        self.other["uav_0"]["velocity_ned_m_s"][0] = .1000001
        self.assertFalse(self.compare()["within_tolerances"]["velocity_l2_m_s"])

    def test_quaternion_sign_equivalence(self):
        self.other["uav_0"]["quaternion_wxyz"] = [-1., 0., 0., 0.]
        self.assertEqual(self.compare()["raw_coordinate_differences"]["attitude_geodesic_rad"], 0)

    def test_analytic_attitude_angle(self):
        for angle in (1e-9, .2, math.pi):
            self.other["uav_0"]["quaternion_wxyz"] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
            self.assertAlmostEqual(self.compare()["raw_coordinate_differences"]["attitude_geodesic_rad"], angle)

    def test_normalizing_for_distance_does_not_repair_bad_norm(self):
        self.other["uav_0"]["quaternion_wxyz"] = [2., 0., 0., 0.]
        result = self.compare()
        self.assertEqual(result["raw_coordinate_differences"]["attitude_geodesic_rad"], 0)
        self.assertFalse(result["within_tolerances"]["quaternion_norm_error"])

    def test_half_turn_geodesic_is_never_greater_than_pi(self):
        self.state["uav_0"]["quaternion_wxyz"] = [math.sqrt(.5), math.sqrt(.5), 0, 0]
        self.other["uav_0"]["quaternion_wxyz"] = [math.sqrt(.5), -math.sqrt(.5), 0, 0]
        config = spec()
        config["tolerances"]["attitude_geodesic_rad"] = math.pi
        result = compare_states(self.state, self.other, config)["uav_0"]
        self.assertAlmostEqual(result["raw_coordinate_differences"]["attitude_geodesic_rad"], math.pi)
        self.assertTrue(result["within_tolerances"]["attitude_geodesic_rad"])

    def test_angular_velocity_compared(self):
        self.other["uav_0"]["angular_velocity_rad_s"][1] = .2
        self.assertFalse(self.compare()["within_tolerances"]["angular_velocity_l2_rad_s"])

    def test_reset_quality_and_modes_not_hidden_by_numerical_agreement(self):
        for key in ("reset_counter", "quality", "nav_state"):
            self.other = deepcopy(self.state)
            self.other["uav_0"][key] += 1
            result = self.compare()
            self.assertTrue(result["all_numeric_checks_passed"])
            self.assertFalse(result["discrete_equal"][key])

    def test_no_cross_session_timestamp_subtraction(self):
        self.other["uav_0"]["timestamp_us"] += 10**18
        self.other["uav_0"]["timestamp_sample_us"] += 10**18
        self.assertTrue(self.compare()["all_numeric_checks_passed"])

    def test_malformed_nonfinite_boolean_zero_and_unknown_frame(self):
        for key, value in (("position_ned_m", [False, 0, 0]), ("velocity_ned_m_s", [math.nan, 0, 0]),
                           ("quaternion_wxyz", [0, 0, 0, 0]), ("angular_velocity_rad_s", [0, 0]),
                           ("pose_frame", 2), ("quality", True), ("failsafe", 0), ("arming_state", 2)):
            self.other = deepcopy(self.state)
            self.other["uav_0"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_agents_never_permuted_to_improve_match(self):
        self.state["uav_0"]["position_ned_m"] = [5, 0, 0]
        self.other = {"uav_0": deepcopy(self.state["uav_1"]), "uav_1": deepcopy(self.state["uav_0"])}
        self.assertFalse(self.compare()["all_numeric_checks_passed"])

    def test_both_agent_renaming_is_symmetric_and_inputs_immutable(self):
        self.other["uav_1"]["velocity_ned_m_s"][0] = .2
        before = deepcopy([self.state, self.other])
        result = compare_states(self.state, self.other, spec())
        swapped = compare_states({"uav_0": self.state["uav_1"], "uav_1": self.state["uav_0"]},
                                 {"uav_0": self.other["uav_1"], "uav_1": self.other["uav_0"]}, spec())
        self.assertEqual(result["uav_0"], swapped["uav_1"])
        self.assertEqual(result["uav_1"], swapped["uav_0"])
        self.assertEqual(before, [self.state, self.other])

    def test_tolerances_must_be_explicit_finite_nonnegative_and_bounded(self):
        for value in (-1, True, math.inf, "0.1"):
            config = spec()
            config["tolerances"]["position_l2_m"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_spec(config)
        for key, value in (("quaternion_norm_error", 1), ("attitude_geodesic_rad", math.pi + .1)):
            config = spec()
            config["tolerances"][key] = value
            with self.assertRaises(ValueError):
                validate_spec(config)
        config = spec()
        del config["tolerances"]["position_l2_m"]
        with self.assertRaises(ValueError):
            validate_spec(config)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = capture(self.base / "evidence")

    def rewrite_manifest(self, change):
        path = self.root / "manifest.json"
        value = json.loads(path.read_bytes())
        change(value)
        path.write_bytes(canonical(value))

    def rewrite_rows(self, change):
        path = self.root / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        change(rows)
        path.write_bytes(b"\n".join(canonical(r) for r in rows) + b"\n")

    def compare(self):
        return compare_startup(self.root, 0, self.root, 0, spec())

    def test_good_clocks_and_identical_numbers_still_ineligible(self):
        result = self.compare()
        self.assertEqual(result["left"]["quality_issues"], [])
        self.assertTrue(result["all_numeric_checks_passed"])
        self.assertTrue(result["same_selected_source"])
        for key in ("start_matched", "physical_state_comparable", "common_coordinate_reference_verified",
                    "candidate_outcomes_emitted", "formal_benchmark_result"):
            self.assertFalse(result[key])
        self.assertIn("controller_and_estimator_internal_state", result["missing_requirements"])
        self.assertEqual(result["left"]["source"]["verified_artifact_files"], 16)
        self.assertFalse(result["left"]["source"]["artifact_semantics_decoded"])

    def test_v1_local_complete_is_not_whole_session_quality(self):
        root = capture(self.base / "old", version=1)
        value = inspect_startup(root, 0)
        self.assertIn("whole_session_clock_quality_unavailable", value["quality_issues"])
        self.assertIsNone(value["session_clock_quality"])

    def test_drift_outside_capture_stays_rejected(self):
        folder = self.base / "drift"
        folder.mkdir()
        root = capture(folder / "evidence", drift=True)
        value = inspect_startup(root, 0)
        self.assertFalse(value["capture_clock_report"]["issues"])
        self.assertIn("session_clock_quality_failed", value["quality_issues"])
        self.assertIn("source_capture_not_complete", value["quality_issues"])

    def test_capture_clock_drift_not_hidden_by_saved_report(self):
        self.rewrite_rows(lambda rows: rows[-1]["host_clocks"]["readings"]["raw_ns"].update(value_ns=20_000_000_000))
        self.assertIn("capture_clock_quality_failed", self.compare()["left"]["quality_issues"])

    def test_explicit_indices_no_nearest_frame_search(self):
        result = compare_startup(self.root, 0, self.root, 1, spec())
        self.assertEqual(result["right"]["source"]["selected_index"], 1)
        self.assertFalse(result["same_selected_source"])
        for index in (-1, 2, True):
            with self.assertRaises(ValueError):
                inspect_startup(self.root, index)

    def test_all_artifacts_including_unselected_row_verified(self):
        path = self.root / "frame_001/uav_1_rgb.pb"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash/length"):
            self.compare()

    def test_session_clock_artifact_hash_verified(self):
        (self.base / "startup_host_clocks.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash/length"):
            self.compare()

    def test_path_escape_windows_and_posix_rejected(self):
        original = (self.root / "samples.jsonl").read_bytes()
        for name in ("../outside", "/absolute", "C:/secret", "frame_000\\uav_0_rgb.pb"):
            (self.root / "samples.jsonl").write_bytes(original)
            self.rewrite_rows(lambda rows: rows[0]["artifacts"]["uav_0_rgb.pb"].update(path=name))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.compare()

    def test_duplicate_json_key_rejected(self):
        (self.root / "manifest.json").write_text('{"complete":false,"complete":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compare()

    def test_record_count_mismatch_and_unsupported_claims_rejected(self):
        original = (self.root / "manifest.json").read_bytes()
        for key, value in (("captured_frames", 3), ("start_matched", True), ("missing_state", []),
                           ("schema", "unknown"), ("complete", 1)):
            (self.root / "manifest.json").write_bytes(original)
            self.rewrite_manifest(lambda m: m.update({key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()

    def test_nonsequential_rows_and_repeated_stamps_rejected(self):
        original = (self.root / "samples.jsonl").read_bytes()
        for key, value in (("index", 4), ("image_stamp_ns", 1_000_000_000)):
            (self.root / "samples.jsonl").write_bytes(original)
            self.rewrite_rows(lambda rows: rows[1].update({key: value}))
            with self.assertRaises(ValueError):
                self.compare()

    def test_saved_quality_flag_cannot_replace_recomputation(self):
        self.rewrite_manifest(lambda m: m["session_clock_quality"].update(quality_passed=False))
        self.assertIn("saved_session_quality_inconsistent", self.compare()["left"]["quality_issues"])

    def test_rebound_hash_with_forged_stable_report_still_rejected(self):
        path = self.base / "startup_host_clocks.json"
        value = json.loads(path.read_bytes())
        value["samples"][-1]["readings"]["raw_ns"]["value_ns"] += 20_000_000
        raw = canonical(value)
        path.write_bytes(raw)
        self.rewrite_manifest(lambda m: m["session_clock_quality"]["artifact"].update(
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        issues = self.compare()["left"]["quality_issues"]
        self.assertIn("session_clock_quality_failed", issues)
        self.assertIn("saved_session_quality_inconsistent", issues)

    def test_stable_but_unrelated_clock_window_not_used(self):
        path = self.base / "startup_host_clocks.json"
        value = json.loads(path.read_bytes())
        for sample in value["samples"]:
            sample["sample_monotonic_ns"] += 1_000_000_000
            for reading in sample["readings"].values():
                for key in ("value_ns", "before_ns", "after_ns"):
                    reading[key] += 1_000_000_000
        raw = canonical(value)
        path.write_bytes(raw)
        self.rewrite_manifest(lambda m: m["session_clock_quality"]["artifact"].update(
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        issues = self.compare()["left"]["quality_issues"]
        self.assertNotIn("session_clock_quality_failed", issues)
        self.assertIn("capture_session_clock_binding_unverified", issues)

    def test_repeated_odometry_sample_not_independent_evidence(self):
        self.rewrite_rows(lambda rows: rows[1]["state"]["uav_1"].update(timestamp_sample_us=1000))
        with self.assertRaisesRegex(ValueError, "Nonadvancing odometry"):
            self.compare()

    def test_unrelated_session_window_or_origin_environment_not_accepted(self):
        self.rewrite_manifest(lambda m: m["environment"].update(boot_id="other-boot"))
        self.assertIn("capture_session_clock_binding_unverified", self.compare()["left"]["quality_issues"])

    def test_different_scene_not_converted_to_a_match(self):
        folder = self.base / "second"
        folder.mkdir()
        other = capture(folder / "evidence")
        path = other / "manifest.json"
        data = json.loads(path.read_bytes())
        data["scene_sha256"] = "b" * 64
        path.write_bytes(canonical(data))
        result = compare_startup(self.root, 0, other, 0, spec())
        self.assertFalse(result["scene_hash_equal"])
        self.assertTrue(result["all_numeric_checks_passed"])
        self.assertFalse(result["start_matched"])

    def test_cli_exit_two_exclusive_output_and_sources_immutable(self):
        config, output = self.base / "spec.json", self.base / "report.json"
        config.write_bytes(canonical(spec()))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.base.rglob("*") if p.is_file()}
        command = [sys.executable, "-B", "-m", "benchmark.px4_gazebo.model_handoff.start_comparison",
                   "--left", str(self.root), "--left-index", "0", "--right", str(self.root),
                   "--right-index", "0", "--spec", str(config), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 2, first.stderr)
        raw = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("FileExistsError", second.stderr)
        self.assertEqual(raw, output.read_bytes())
        self.assertEqual(before, {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})


if __name__ == "__main__":
    unittest.main()
