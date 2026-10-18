"""Bounded, no-flight host-clock diagnostic. Never changes system clocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

from .trace_quality import clock_pair_diagnostics

PROBE_SCHEMA = "aerowam-host-clock-probe/1"
LIMIT_NS = 10_000_000


def sample_clocks() -> dict:
    start = time.monotonic_ns()
    readings = {}
    for name, clock in (("unix_ns", "CLOCK_REALTIME"), ("raw_ns", "CLOCK_MONOTONIC_RAW"),
                        ("boottime_ns", "CLOCK_BOOTTIME")):
        if hasattr(time, "clock_gettime_ns") and hasattr(time, clock):
            before = time.monotonic_ns()
            value = time.clock_gettime_ns(getattr(time, clock))
            after = time.monotonic_ns()
        elif name == "unix_ns":
            before = time.monotonic_ns()
            value = time.time_ns()
            after = time.monotonic_ns()
        else:
            continue
        readings[name] = {"value_ns": value, "before_ns": before, "after_ns": after}
    return {"sample_monotonic_ns": start, "readings": readings}


def analyze(samples: list[dict]) -> dict:
    if len(samples) < 2:
        raise ValueError("At least two clock samples required")
    if any(b["sample_monotonic_ns"] <= a["sample_monotonic_ns"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Non-increasing probe clock")
    metrics = {}
    events = []
    for name in ("unix_ns", "raw_ns", "boottime_ns"):
        if not all(name in sample["readings"] for sample in samples):
            metrics[name] = {"available": False}
            continue
        stream = []
        for sample in samples:
            reading = sample["readings"][name]
            if not all(isinstance(reading[key], int) for key in ("value_ns", "before_ns", "after_ns")):
                raise ValueError("Invalid clock reading")
            if reading["after_ns"] < reading["before_ns"]:
                raise ValueError("Backward clock read bracket")
            stream.append({"received_unix_ns": reading["value_ns"],
                           "received_monotonic_ns": reading["before_ns"],
                           "clock_read_end_monotonic_ns": reading["after_ns"]})
        changes, widths = clock_pair_diagnostics(stream)
        metrics[name] = {"available": True, "max_offset_change_outside_bounds_ms": max(changes),
                         "max_read_width_ms": max(widths)}
        for index, change in enumerate(changes):
            if change * 1e6 > LIMIT_NS:
                events.append({"clock": name, "from_sample": index, "to_sample": index + 1,
                               "minimum_offset_change_ms": change})
    issues = []
    if events:
        issues.append("clock_offset_inconsistency")
    if any(metric.get("max_read_width_ms", 0) * 1e6 > LIMIT_NS for metric in metrics.values()):
        issues.append("clock_read_uncertainty")
    gaps = [b["sample_monotonic_ns"] - a["sample_monotonic_ns"] for a, b in zip(samples, samples[1:])]
    return {"samples": len(samples), "elapsed_monotonic_s":
            (samples[-1]["sample_monotonic_ns"] - samples[0]["sample_monotonic_ns"]) / 1e9,
            "max_poll_gap_ms": max(gaps) / 1e6, "metrics": metrics, "events": events,
            "issues": issues, "interpretation":
            "Relative clock evidence only; no proof of OS cause, simulator alignment or verified actuation."}


def environment() -> dict:
    result = {"platform": platform.platform(), "clock_names": ["monotonic", "realtime", "monotonic_raw", "boottime"]}
    for name, path in (("boot_id", "/proc/sys/kernel/random/boot_id"),
                       ("clocksource", "/sys/devices/system/clocksource/clocksource0/current_clocksource")):
        try:
            result[name] = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            result[name] = None
    return result


def run_probe(path: Path, *, duration_s: float = 20., interval_ms: float = 50.) -> dict:
    if (not math.isfinite(duration_s) or not 1 <= duration_s <= 60
            or not math.isfinite(interval_ms) or not 10 <= interval_ms <= 1000):
        raise ValueError("duration must be 1..60 s; interval must be 10..1000 ms")
    # Reserve without overwriting any earlier diagnostic evidence.
    with path.open("x", encoding="utf-8") as output:
        info = environment()
        samples = []
        deadline = time.monotonic() + duration_s
        interrupted = False
        try:
            while True:
                samples.append(sample_clocks())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, interval_ms / 1000))
        except KeyboardInterrupt:
            interrupted = True
        report = analyze(samples) if len(samples) >= 2 else {"issues": ["insufficient_samples"]}
        output.write(json.dumps({"schema": PROBE_SCHEMA, "environment": info, "interrupted": interrupted,
                                 "samples": samples, "report": report}, allow_nan=False) + "\n")
    if interrupted:
        raise KeyboardInterrupt
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=20.)
    parser.add_argument("--interval-ms", type=float, default=50.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-stable", action="store_true", help="exit 2 for clock inconsistencies/uncertainty")
    args = parser.parse_args()
    try:
        report = run_probe(args.output, duration_s=args.duration_s, interval_ms=args.interval_ms)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.require_stable and report["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
