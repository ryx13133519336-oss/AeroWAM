"""No-flight drift/uncertainty regressions; no system clock writes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .clock_probe import ANALYSIS_PROFILE, PROBE_SCHEMA, analyze, reanalyze_probe, run_probe


def samples(offsets, *, interval_ns=50_000_000, width_ns=100):
    epoch = 1_790_000_000_000_000_000
    rows = []
    for i, offset in enumerate(offsets):
        now = 1_000_000_000 + i * interval_ns
        rows.append({"sample_monotonic_ns": now, "readings": {
            name: {"value_ns": now + epoch + (offset if name == "raw_ns" else 0),
                   "before_ns": now, "after_ns": now + width_ns}
            for name in ("unix_ns", "raw_ns", "boottime_ns")}})
    return rows


class ClockProbeTests(unittest.TestCase):
    def test_constant_offsets_and_large_epoch_are_exact(self):
        report = analyze(samples([0] * 4))
        self.assertEqual(report["analysis_profile"], ANALYSIS_PROFILE)
        self.assertFalse(report["issues"])
        metric = report["metrics"]["raw_ns"]
        self.assertEqual(metric["minimum_offset_excursion_ms"], 0.)
        self.assertEqual(metric["net_offset_change_ms_bounds"], [-.0001, .0001])

    def test_slow_drift_below_every_adjacent_limit_is_rejected(self):
        report = analyze(samples([i * 1_000_000 for i in range(21)]))
        self.assertFalse(report["events"])
        self.assertEqual(report["issues"], ["clock_offset_excursion"])
        self.assertGreater(report["metrics"]["raw_ns"]["minimum_offset_excursion_ms"], 19.99)

    def test_cadence_does_not_hide_same_observed_excursion(self):
        dense = analyze(samples([i * 1_000_000 for i in range(21)]))
        sparse = analyze(samples([0, 20_000_000], interval_ns=1_000_000_000))
        self.assertTrue(dense["issues"] and sparse["issues"])
        self.assertEqual(dense["metrics"]["raw_ns"]["minimum_offset_excursion_ms"],
                         sparse["metrics"]["raw_ns"]["minimum_offset_excursion_ms"])

    def test_roundtrip_drift_is_not_hidden_by_equal_endpoints(self):
        report = analyze(samples([i * 1_000_000 for i in range(21)] +
                                 [i * 1_000_000 for i in range(19, -1, -1)]))
        metric = report["metrics"]["raw_ns"]
        self.assertIn("clock_offset_excursion", report["issues"])
        self.assertGreater(metric["minimum_offset_excursion_ms"], 19.99)
        self.assertLess(metric["net_offset_change_ms_bounds"][1], .001)

    def test_negative_drift_and_signed_rate(self):
        report = analyze(samples([0, -20_000_000], interval_ns=1_000_000_000))
        metric = report["metrics"]["raw_ns"]
        self.assertLess(metric["net_offset_change_ms_bounds"][1], -19.99)
        self.assertAlmostEqual(sum(metric["net_relative_rate_ppm_bounds"]) / 2, -20_000, places=2)
        self.assertIn("clock_offset_excursion", report["issues"])

    def test_positive_rate_bounds(self):
        metric = analyze(samples([0, 20_000_000], interval_ns=1_000_000_000))["metrics"]["raw_ns"]
        low, high = metric["net_relative_rate_ppm_bounds"]
        self.assertLess(low, 20_000)
        self.assertGreater(high, 20_000)

    def test_broad_bracket_is_uncertain_not_proof_of_excursion(self):
        rows = samples([0, 20_000_000], width_ns=50_000_000)
        report = analyze(rows)
        self.assertEqual(report["issues"], ["clock_read_uncertainty"])
        self.assertEqual(report["metrics"]["raw_ns"]["minimum_offset_excursion_ms"], 0.)
        self.assertIsNone(report["metrics"]["raw_ns"]["net_relative_rate_ppm_bounds"])

    def test_exact_limit_not_relaxed_and_one_ns_excess_detected(self):
        # 100 ns read uncertainty: the lower bound is exactly 10 ms here.
        self.assertFalse(analyze(samples([0, 10_000_100]))["issues"])
        self.assertIn("clock_offset_excursion", analyze(samples([0, 10_000_101]))["issues"])

    def test_missing_clock_is_not_fabricated(self):
        rows = samples([0, 0])
        for row in rows:
            del row["readings"]["raw_ns"]
        self.assertEqual(analyze(rows)["metrics"]["raw_ns"], {"available": False})

    def test_invalid_readings_and_order_rejected(self):
        for key, value in (("value_ns", True), ("before_ns", 1.), ("after_ns", -1)):
            rows = samples([0, 0])
            rows[1]["readings"]["raw_ns"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze(rows)
        rows = samples([0, 0])
        rows[1]["sample_monotonic_ns"] = rows[0]["sample_monotonic_ns"]
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_input_not_mutated(self):
        rows = samples([0, 20_000_000])
        saved = deepcopy(rows)
        analyze(rows)
        self.assertEqual(rows, saved)

    def test_separate_reanalysis_retains_source_and_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "old.json", Path(directory) / "new.json"
            source.write_text(json.dumps({"schema": PROBE_SCHEMA, "samples": samples([i*1_000_000 for i in range(21)]),
                                         "report": {"issues": []}, "interrupted": False}), encoding="utf-8")
            original = source.read_bytes()
            report = reanalyze_probe(source, output)
            self.assertEqual(report["issues"], ["clock_offset_excursion"])
            saved = json.loads(output.read_text())
            self.assertEqual(saved["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(saved["source_analysis_profile"], "legacy-adjacent-only")
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                reanalyze_probe(source, output)
            with self.assertRaises(FileExistsError):
                reanalyze_probe(source, source)
            self.assertEqual(source.read_bytes(), original)

    def test_reanalysis_rejects_wrong_schema_without_output(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "wrong.json", Path(directory) / "new.json"
            source.write_text(json.dumps({"schema": "other", "samples": samples([0, 0])}), encoding="utf-8")
            with self.assertRaises(ValueError):
                reanalyze_probe(source, output)
            self.assertFalse(output.exists())

    def test_cli_reanalysis_exit_and_sampling_options(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "old.json", Path(directory) / "new.json"
            source.write_text(json.dumps({"schema": PROBE_SCHEMA, "samples": samples([i*1_000_000 for i in range(21)])}), encoding="utf-8")
            command = [sys.executable, "-B", "-m", "benchmark.px4_gazebo.model_handoff.clock_probe",
                       "--input", str(source), "--output", str(output), "--require-stable"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15.)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("clock_offset_excursion", json.loads(result.stdout)["issues"])
            self.assertTrue(output.exists())
            invalid = subprocess.run(command + ["--duration-s", "1"], capture_output=True, text=True, timeout=15.)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("cannot be combined", invalid.stderr)

    def test_live_probe_records_start_end_environment_without_sleep(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            base = "benchmark.px4_gazebo.model_handoff.clock_probe."
            before = {"boot_id": "same-boot", "uptime": "1 2", "clocksource": "tsc"}
            after = {**before, "uptime": "2 3"}
            with patch(base + "time.monotonic", side_effect=(0., .5, 1.)), \
                 patch(base + "time.sleep") as sleep, \
                 patch(base + "sample_clocks", side_effect=samples([0, 0])), \
                 patch(base + "environment", side_effect=(before, after)):
                report = run_probe(path, duration_s=1.)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["environment"], before)
            self.assertEqual(saved["environment_after"], after)
            self.assertEqual(saved["report"]["analysis_profile"], ANALYSIS_PROFILE)
            self.assertFalse(report["issues"])
            sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
