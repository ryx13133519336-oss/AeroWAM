"""Synthetic read-only/kernel ABI and clock-gate tests; no clock changes."""

from copy import deepcopy
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .candidate_scoring import canonical
from .clock_kernel_probe import (SCHEMA, KernelReader, _Timex, _supported_abi,
                                  inspect, run, service_context, summarize)
from .clock_probe_self_test import samples

MODULE = "benchmark.px4_gazebo.model_handoff.clock_kernel_probe."
ENV = {"boot_id": "synthetic-boot", "clocksource": "tsc", "uptime": "1 2"}


def records(offsets=None):
    clocks = samples([0] * 4 if offsets is None else offsets)
    return [{"host_clocks": row, "kernel": {"available": True, "requested_modes": 0,
        "result": 0, "before_ns": row["sample_monotonic_ns"] + 200,
        "after_ns": row["sample_monotonic_ns"] + 300, "offset": 0, "offset_unit": "us",
        "frequency_scaled_ppm": 0, "tick_us": 10000, "status": 0,
        "maxerror_us": 100, "esterror_us": 10}} for row in clocks]


def payload(rows=None):
    rows = records() if rows is None else rows
    return {"schema": SCHEMA, "source_kind": "synthetic_fixture", "duration_s": .15,
        "target_interval_ms": 50., "environment": ENV, "environment_after": ENV,
        "service_before": {"synthetic": True}, "service_after": {"synthetic": True},
        "records": rows, "errors": [], "summary": summarize(rows, ENV, ENV)}


class KernelProbeTests(unittest.TestCase):
    def test_every_query_receives_fresh_zero_bytes_and_readonly_modes(self):
        seen = []
        def query(pointer):
            seen.append(ctypes.string_at(pointer, ctypes.sizeof(_Timex)))
            value = ctypes.cast(pointer, ctypes.POINTER(_Timex)).contents
            value.offset, value.freq, value.tick = -12, 65536, 10000
            return 0
        with patch(MODULE + "_supported_abi"), patch(MODULE + "ctypes.CDLL", return_value=SimpleNamespace(adjtimex=query)):
            reader = KernelReader()
            a, b = reader(), reader()
        self.assertEqual(seen, [bytes(ctypes.sizeof(_Timex))] * 2)
        self.assertEqual(a["requested_modes"], 0)
        self.assertEqual(b["offset"], -12)
        self.assertEqual(b["frequency_scaled_ppm"], 65536)

    def test_unsupported_platform_refuses_before_any_library_query(self):
        with patch(MODULE + "sys.platform", "win32"), patch(MODULE + "ctypes.CDLL") as library:
            with self.assertRaisesRegex(RuntimeError, "supported Linux"):
                KernelReader()
            library.assert_not_called()

    def test_layout_mismatch_refuses_query(self):
        with patch(MODULE + "sys.platform", "linux"), patch(MODULE + "platform.machine", return_value="x86_64"), \
                patch(MODULE + "ctypes.sizeof", return_value=1):
            with self.assertRaises(RuntimeError):
                _supported_abi()

    def test_errno_is_failure_without_a_write_retry(self):
        calls = []
        def query(pointer):
            calls.append(1)
            ctypes.set_errno(1)
            return -1
        with patch(MODULE + "_supported_abi"), patch(MODULE + "ctypes.CDLL", return_value=SimpleNamespace(adjtimex=query)):
            reader = KernelReader()
            with self.assertRaises(OSError):
                reader()
        self.assertEqual(calls, [1])

    def test_nanosecond_flag_controls_offset_units(self):
        def query(pointer):
            value = ctypes.cast(pointer, ctypes.POINTER(_Timex)).contents
            value.status, value.tick = 0x2000, 10000
            return 0
        with patch(MODULE + "_supported_abi"), patch(MODULE + "ctypes.CDLL", return_value=SimpleNamespace(adjtimex=query)):
            self.assertEqual(KernelReader()()["offset_unit"], "ns")

    def test_kernel_frequency_scale_is_not_interpreted_as_microseconds(self):
        rows = records()
        rows[0]["kernel"]["frequency_scaled_ppm"] = -65536
        rows[1]["kernel"]["frequency_scaled_ppm"] = 65536 * 500
        report = summarize(rows, ENV, ENV)
        self.assertEqual(report["kernel_frequency_ppm_range"], [-1., 500.])
        self.assertEqual(report["kernel_tick_us_range"], [10000, 10000])

    def test_matching_frequency_never_detrends_or_waives_10ms_excursion(self):
        rows = records([i * 25_000 for i in range(422)])  # 500 ppm over >21s.
        for row in rows:
            row["kernel"]["frequency_scaled_ppm"] = -500 * 65536
        report = summarize(rows, ENV, ENV)
        self.assertFalse(report["quality_passed"])
        self.assertIn("clock_offset_excursion", report["issues"])
        self.assertFalse(report["clock_correction_applied"])
        self.assertFalse(report["cause_identified"])

    def test_equal_endpoints_do_not_hide_roundtrip_excursion(self):
        report = summarize(records([0, 20_000_000, 0]), ENV, ENV)
        self.assertIn("clock_offset_excursion", report["issues"])

    def test_kernel_unsynchronized_and_time_error_remain_explicit(self):
        for field, value in (("result", 5), ("status", 0x40)):
            rows = records()
            rows[1]["kernel"][field] = value
            report = summarize(rows, ENV, ENV)
            self.assertIn("kernel_unsynchronized_observed", report["issues"])
            self.assertFalse(report["quality_passed"])
            self.assertTrue(report["clock_quality"]["quality_passed"])

    def test_missing_kernel_never_fabricates_context(self):
        rows = records()
        rows[0]["kernel"] = {"available": False, "error": "synthetic query error"}
        report = summarize(rows, ENV, ENV)
        self.assertIn("kernel_context_unavailable", report["issues"])
        self.assertEqual(report["kernel_samples"], 3)

    def test_nonzero_modes_boolean_numbers_and_wrong_units_rejected(self):
        for field, value in (("requested_modes", 1), ("result", True), ("status", -1),
                             ("offset_unit", "ns"), ("frequency_scaled_ppm", 0.5)):
            rows = records()
            rows[0]["kernel"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(rows, ENV, ENV)

    def test_overlapping_or_backward_kernel_brackets_rejected(self):
        for before, after in ((1, 0), (1, 2)):
            rows = records()
            rows[0]["kernel"].update(before_ns=before, after_ns=after)
            with self.assertRaises(ValueError):
                summarize(rows, ENV, ENV)

    def test_slow_kernel_read_is_not_precise_context(self):
        rows = records()
        rows[-1]["kernel"]["after_ns"] += 11_000_000
        self.assertIn("kernel_read_uncertainty", summarize(rows, ENV, ENV)["issues"])

    def test_boot_change_is_not_same_session_evidence(self):
        after = {**ENV, "boot_id": "other"}
        self.assertIn("clock_environment_unverified", summarize(records(), ENV, after)["issues"])

    def test_missing_raw_clock_keeps_existing_gate_rejection(self):
        rows = records()
        del rows[0]["host_clocks"]["readings"]["raw_ns"]
        self.assertIn("required_host_clock_unavailable", summarize(rows, ENV, ENV)["issues"])

    def test_existing_gap_and_monitor_error_checks_are_preserved(self):
        rows = records()
        rows[-1]["host_clocks"]["sample_monotonic_ns"] += 300_000_000
        for reading in rows[-1]["host_clocks"]["readings"].values():
            for key in ("value_ns", "before_ns", "after_ns"):
                reading[key] += 300_000_000
        rows[-1]["kernel"]["before_ns"] += 300_000_000
        rows[-1]["kernel"]["after_ns"] += 300_000_000
        report = summarize(rows, ENV, ENV, ["synthetic worker failure"])
        self.assertIn("clock_poll_gap", report["issues"])
        self.assertIn("clock_monitor_error", report["issues"])

    def test_good_probe_has_no_startup_alignment_or_flight_claim(self):
        rows = records()
        before = deepcopy(rows)
        report = summarize(rows, ENV, ENV)
        self.assertTrue(report["quality_passed"])
        self.assertEqual(rows, before)
        for name in ("startup_quality_verified", "exposure_alignment_verified", "start_matching_verified", "flight_ready"):
            self.assertIs(report[name], False)

    def test_saved_evidence_inspection_uses_no_kernel_or_service_calls(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.json", Path(directory) / "check.json"
            source.write_bytes(canonical(payload()))
            original = source.read_bytes()
            with patch(MODULE + "KernelReader") as reader, patch(MODULE + "service_context") as service:
                report = inspect(source, output)
                reader.assert_not_called()
                service.assert_not_called()
            self.assertTrue(report["saved_summary_consistent"])
            self.assertEqual(report["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                inspect(source, output)
            with self.assertRaises(FileExistsError):
                inspect(source, source)

    def test_changed_saved_summary_is_detected_without_rewriting_it(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.json", Path(directory) / "check.json"
            data = payload()
            data["summary"]["kernel_frequency_ppm_range"] = [0, 1]
            source.write_bytes(canonical(data))
            self.assertFalse(inspect(source, output)["saved_summary_consistent"])

    def test_live_capture_query_failure_is_recorded_and_not_retried(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            with patch(MODULE + "KernelReader") as factory, patch(MODULE + "environment", return_value=ENV), \
                    patch(MODULE + "service_context", return_value={"synthetic": True}):
                factory.return_value.side_effect = OSError("synthetic query failure")
                report = run(output, duration_s=1.)
                self.assertEqual(factory.return_value.call_count, 1)
            data = json.loads(output.read_bytes())
            self.assertFalse(report["quality_passed"])
            self.assertEqual(len(data["records"]), 1)
            self.assertIn("kernel_context_unavailable", report["issues"])

    def test_invalid_live_budget_does_not_create_file_or_query_kernel(self):
        with TemporaryDirectory() as directory, patch(MODULE + "KernelReader") as reader:
            output = Path(directory) / "raw.json"
            for value in (True, 0., 61., float("nan")):
                with self.assertRaises(ValueError):
                    run(output, duration_s=value)
            reader.assert_not_called()
            self.assertFalse(output.exists())

    def test_service_query_only_uses_bounded_systemctl_show(self):
        with patch(MODULE + "subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as query:
            service_context()
        args, kwargs = query.call_args
        self.assertEqual(args[0][:3], ["systemctl", "show", "systemd-timesyncd"])
        self.assertEqual(kwargs["timeout"], 2)
        self.assertNotIn("shell", kwargs)

    def test_live_capture_success_is_recomputable_and_never_overwrites(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            rows = records()[:2]
            with patch(MODULE + "KernelReader") as factory, patch(MODULE + "environment", return_value=ENV), \
                    patch(MODULE + "service_context", return_value={"synthetic": True}), \
                    patch(MODULE + "sample_clocks", side_effect=[r["host_clocks"] for r in rows]), \
                    patch(MODULE + "time.monotonic", side_effect=[0., 0., 1.1]), patch(MODULE + "time.sleep"):
                factory.return_value.side_effect = [r["kernel"] for r in rows]
                report = run(output, duration_s=1.)
                self.assertTrue(report["quality_passed"])
            self.assertTrue(inspect(output, Path(directory) / "check.json")["saved_summary_consistent"])
            original = output.read_bytes()
            with patch(MODULE + "KernelReader") as factory:
                with self.assertRaises(FileExistsError):
                    run(output, duration_s=1.)
                factory.return_value.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_interruption_preserves_partial_raw_records_and_refuses_quality(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            row = records()[0]
            with patch(MODULE + "KernelReader") as factory, patch(MODULE + "environment", return_value=ENV), \
                    patch(MODULE + "service_context", return_value={"synthetic": True}), \
                    patch(MODULE + "sample_clocks", return_value=row["host_clocks"]), \
                    patch(MODULE + "time.monotonic", return_value=0.), \
                    patch(MODULE + "time.sleep", side_effect=KeyboardInterrupt):
                factory.return_value.return_value = row["kernel"]
                report = run(output, duration_s=1.)
            saved = json.loads(output.read_bytes())
            self.assertEqual(saved["errors"], ["KeyboardInterrupt"])
            self.assertEqual(len(saved["records"]), 1)
            self.assertIn("clock_monitor_error", report["issues"])
            self.assertFalse(report["quality_passed"])

    def test_cli_saved_good_bad_and_duplicate_json_rejection(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.json", Path(directory) / "check.json"
            source.write_bytes(canonical(payload()))
            command = [sys.executable, "-B", "-m", MODULE[:-1], "--input", str(source), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            bad = payload(records([0, 20_000_000, 0]))
            source.write_bytes(canonical(bad))
            command[-1] = str(Path(directory) / "bad_check.json")
            self.assertEqual(subprocess.run(command, capture_output=True, timeout=15).returncode, 2)
            source.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            command[-1] = str(Path(directory) / "no_check.json")
            self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=15).returncode, 0)
            self.assertFalse(Path(command[-1]).exists())


if __name__ == "__main__":
    unittest.main()
