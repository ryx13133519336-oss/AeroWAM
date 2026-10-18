"""Read-only Linux clock/kernel context; never adjusts clocks or gate thresholds.

adjtimex is called ONLY with a fresh zeroed structure (modes=0). Live collection
supports Linux x86_64 LP64; saved evidence can be inspected on Windows too.
"""

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from .adapter_manifest import _load
from .candidate_scoring import canonical
from .clock_probe import environment, sample_clocks
from .language_tasks import exact_keys
from .startup_clock import session_quality

SCHEMA = "aerowam-clock-kernel-probe/1-draft"
SCOPE = "read_only_probe_window_not_simulator_startup"


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _Timex(ctypes.Structure):
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
        ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
        ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
        ("time", _Timeval), ("tick", ctypes.c_long), ("ppsfreq", ctypes.c_long),
        ("jitter", ctypes.c_long), ("shift", ctypes.c_int), ("stabil", ctypes.c_long),
        ("jitcnt", ctypes.c_long), ("calcnt", ctypes.c_long), ("errcnt", ctypes.c_long),
        ("stbcnt", ctypes.c_long), ("tai", ctypes.c_int), ("padding", ctypes.c_int * 11)]


def _supported_abi():
    if (sys.platform != "linux" or platform.machine() != "x86_64"
            or ctypes.sizeof(ctypes.c_long) != 8 or ctypes.sizeof(_Timex) != 208
            or any(getattr(_Timex, field).offset != offset for field, offset in
                   (("modes", 0), ("offset", 8), ("freq", 16), ("status", 40),
                    ("time", 72), ("tick", 88), ("tai", 160)))):
        raise RuntimeError("Live kernel queries require the supported Linux x86_64 LP64 timex ABI")


class KernelReader:
    """No setter, mode argument or CAP_SYS_TIME request exists in this wrapper."""

    def __init__(self):
        _supported_abi()
        self.library = ctypes.CDLL(None, use_errno=True)
        self.query = self.library.adjtimex
        self.query.argtypes = [ctypes.POINTER(_Timex)]
        self.query.restype = ctypes.c_int

    def __call__(self):
        value = _Timex()  # ctypes zero-initializes every byte, including modes.
        before = time.monotonic_ns()
        result = self.query(ctypes.byref(value))
        after = time.monotonic_ns()
        if result < 0:
            number = ctypes.get_errno()
            raise OSError(number, "Read-only adjtimex(modes=0) query failed")
        if value.modes != 0:
            raise RuntimeError("Unexpected returned modes in read-only kernel query")
        return {"available": True, "requested_modes": 0, "result": result,
                "before_ns": before, "after_ns": after, "offset": value.offset,
                "offset_unit": "ns" if value.status & 0x2000 else "us",
                "frequency_scaled_ppm": value.freq, "tick_us": value.tick,
                "status": value.status, "maxerror_us": value.maxerror, "esterror_us": value.esterror}


def service_context():
    """Bounded read-only endpoint observation, not continuous service monitoring."""
    command = ["systemctl", "show", "systemd-timesyncd", "--no-pager",
               "--property=ActiveState,SubState,UnitFileState,InvocationID"]
    before = time.monotonic_ns()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2)
        data = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        data = {"error": f"{type(exc).__name__}:{exc}"}
    return {"command": command, "before_ns": before, "after_ns": time.monotonic_ns(), **data}


def summarize(records, before, after, errors=()):
    if not isinstance(records, list):
        raise ValueError("Explicit clock/kernel sample list required")
    clocks, kernels, unavailable = [], [], []
    previous_kernel_end = -1
    for index, row in enumerate(records):
        exact_keys(row, {"host_clocks", "kernel"})
        clocks.append(row["host_clocks"])
        kernel = row["kernel"]
        if kernel.get("available") is False:
            exact_keys(kernel, {"available", "error"})
            unavailable.append({"index": index, "error": kernel["error"]})
            continue
        exact_keys(kernel, {"available", "requested_modes", "result", "before_ns", "after_ns",
                            "offset", "offset_unit", "frequency_scaled_ppm", "tick_us",
                            "status", "maxerror_us", "esterror_us"})
        if kernel["available"] is not True or any(type(kernel[name]) is not int for name in (
                "requested_modes", "result", "before_ns", "after_ns", "offset",
                "frequency_scaled_ppm", "tick_us", "status", "maxerror_us", "esterror_us")):
            raise ValueError("Invalid raw kernel context types")
        if (kernel["requested_modes"] != 0 or not 0 <= kernel["result"] <= 5
                or kernel["after_ns"] < kernel["before_ns"]
                or kernel["before_ns"] < max(r["after_ns"] for r in row["host_clocks"]["readings"].values())
                or row["host_clocks"]["sample_monotonic_ns"] <= previous_kernel_end
                or kernel["status"] < 0 or kernel["tick_us"] <= 0
                or kernel["maxerror_us"] < 0 or kernel["esterror_us"] < 0
                or kernel["offset_unit"] != ("ns" if kernel["status"] & 0x2000 else "us")):
            raise ValueError("Invalid read-only kernel context/units")
        kernels.append(kernel)
        previous_kernel_end = kernel["after_ns"]
    # The existing gate is recomputed unchanged. In particular, no frequency
    # subtraction, detrending or best-subwindow selection happens here.
    quality = session_quality(clocks, before, after, errors)
    quality["scope"] = SCOPE
    issues = list(quality["issues"])
    if unavailable or len(kernels) != len(records) or not kernels:
        issues.append("kernel_context_unavailable")
    if any(k["result"] == 5 or k["status"] & 0x40 for k in kernels):
        issues.append("kernel_unsynchronized_observed")
    widths = [(k["after_ns"] - k["before_ns"]) / 1e6 for k in kernels]
    if widths and max(widths) > 10:
        issues.append("kernel_read_uncertainty")
    frequencies = [k["frequency_scaled_ppm"] / 65536 for k in kernels]
    ticks = [k["tick_us"] for k in kernels]
    return {"scope": SCOPE, "quality_passed": not issues, "issues": issues,
            "clock_quality": quality, "kernel_samples": len(kernels),
            "kernel_query_errors": unavailable,
            "kernel_frequency_ppm_range": [min(frequencies), max(frequencies)] if frequencies else None,
            "kernel_tick_us_range": [min(ticks), max(ticks)] if ticks else None,
            "kernel_status_values": sorted({k["status"] for k in kernels}),
            "kernel_result_values": sorted({k["result"] for k in kernels}),
            "kernel_max_read_width_ms": max(widths) if widths else None,
            "cause_identified": False, "clock_correction_applied": False,
            "startup_quality_verified": False, "exposure_alignment_verified": False,
            "start_matching_verified": False, "flight_ready": False}


def run(output, *, duration_s=20., interval_ms=50.):
    if (isinstance(duration_s, bool) or not math.isfinite(duration_s) or not 1 <= duration_s <= 60
            or isinstance(interval_ms, bool) or not math.isfinite(interval_ms) or not 10 <= interval_ms <= 250):
        raise ValueError("duration must be 1..60 s; interval must be 10..250 ms")
    reader = KernelReader()  # Refuse unsupported ABI before creating a record.
    with Path(output).open("xb") as stream:
        before, context_before = environment(), service_context()
        records, errors = [], []
        deadline = time.monotonic() + duration_s
        try:
            while True:
                clock = sample_clocks()
                try:
                    kernel = reader()
                except (OSError, RuntimeError) as exc:
                    kernel = {"available": False, "error": f"{type(exc).__name__}:{exc}"}
                records.append({"host_clocks": clock, "kernel": kernel})
                if not kernel["available"]:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, interval_ms / 1000))
        except KeyboardInterrupt:
            errors.append("KeyboardInterrupt")
        after, context_after = environment(), service_context()
        summary = summarize(records, before, after, errors)
        payload = {"schema": SCHEMA, "source_kind": "live_read_only", "duration_s": duration_s,
                   "target_interval_ms": interval_ms, "environment": before, "environment_after": after,
                   "service_before": context_before, "service_after": context_after,
                   "records": records, "errors": errors, "summary": summary}
        stream.write(canonical(payload) + b"\n")
    return summary


def inspect(source, output):
    raw = Path(source).read_bytes()
    payload = _load(raw)
    exact_keys(payload, {"schema", "source_kind", "duration_s", "target_interval_ms", "environment",
                         "environment_after", "service_before", "service_after", "records", "errors", "summary"})
    if payload["schema"] != SCHEMA or payload["source_kind"] not in {"live_read_only", "synthetic_fixture"}:
        raise ValueError("Expected clock/kernel probe evidence")
    result = summarize(payload["records"], payload["environment"], payload["environment_after"], payload["errors"])
    report = {"schema": "aerowam-clock-kernel-inspection/1-draft", "source_sha256": hashlib.sha256(raw).hexdigest(),
              "source_kind": payload["source_kind"], "saved_summary_consistent": result == payload["summary"],
              "summary": result}
    with Path(output).open("xb") as stream:
        stream.write(canonical(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Inspect saved evidence only; no live queries")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--interval-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input is not None:
        if args.duration_s is not None or args.interval_ms is not None:
            parser.error("--input cannot be combined with live sampling options")
        result = inspect(args.input, args.output)
        success = result["saved_summary_consistent"] and result["summary"]["quality_passed"]
    else:
        result = run(args.output, duration_s=20. if args.duration_s is None else args.duration_s,
                     interval_ms=50. if args.interval_ms is None else args.interval_ms)
        success = result["quality_passed"]
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
