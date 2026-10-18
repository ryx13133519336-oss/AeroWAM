"""Bounded, no-flight host-clock diagnostic. Never changes system clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from .trace_quality import clock_pair_diagnostics

PROBE_SCHEMA = "aerowam-host-clock-probe/1"
ANALYSIS_PROFILE = "aerowam-host-clock-analysis/2-draft"
LIMIT_NS = 10_000_000


def offset_excursion(stream: list[dict]) -> dict:
    """Bounds on observed offset excursion and net relative rate, not a cause.

    The largest lower bound minus the smallest upper bound detects slow
    accumulated drift and excursions that return to the original offset.
    Keeping integer differences avoids rounding UNIX epochs in float math.
    """
    lows = [s["received_unix_ns"] - s["clock_read_end_monotonic_ns"] for s in stream]
    highs = [s["received_unix_ns"] - s["received_monotonic_ns"] for s in stream]
    first, last = stream[0], stream[-1]
    dt_min = last["received_monotonic_ns"] - first["clock_read_end_monotonic_ns"]
    dt_max = last["clock_read_end_monotonic_ns"] - first["received_monotonic_ns"]
    delta = last["received_unix_ns"] - first["received_unix_ns"]
    rates = (sorted(((delta - dt_max) * 1e6 / dt_max, (delta - dt_min) * 1e6 / dt_min))
             if dt_min > 0 else None)
    return {"minimum_offset_excursion_ms": max(0, max(lows) - min(highs)) / 1e6,
            "net_offset_change_ms_bounds": [(lows[-1] - highs[0]) / 1e6,
                                             (highs[-1] - lows[0]) / 1e6],
            "net_relative_rate_ppm_bounds": rates}


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
            if not all(type(reading[key]) is int for key in ("value_ns", "before_ns", "after_ns")):
                raise ValueError("Invalid clock reading")
            if reading["after_ns"] < reading["before_ns"]:
                raise ValueError("Backward clock read bracket")
            stream.append({"received_unix_ns": reading["value_ns"],
                           "received_monotonic_ns": reading["before_ns"],
                           "clock_read_end_monotonic_ns": reading["after_ns"]})
        changes, widths = clock_pair_diagnostics(stream)
        metrics[name] = {"available": True, "max_offset_change_outside_bounds_ms": max(changes),
                         "max_read_width_ms": max(widths), **offset_excursion(stream)}
        for index, change in enumerate(changes):
            if change * 1e6 > LIMIT_NS:
                events.append({"clock": name, "from_sample": index, "to_sample": index + 1,
                               "minimum_offset_change_ms": change})
    issues = []
    if events:
        issues.append("clock_offset_inconsistency")
    if any(metric.get("minimum_offset_excursion_ms", 0) * 1e6 > LIMIT_NS for metric in metrics.values()):
        issues.append("clock_offset_excursion")
    if any(metric.get("max_read_width_ms", 0) * 1e6 > LIMIT_NS for metric in metrics.values()):
        issues.append("clock_read_uncertainty")
    gaps = [b["sample_monotonic_ns"] - a["sample_monotonic_ns"] for a, b in zip(samples, samples[1:])]
    return {"analysis_profile": ANALYSIS_PROFILE, "offset_change_limit_ms": LIMIT_NS / 1e6,
            "samples": len(samples), "elapsed_monotonic_s":
            (samples[-1]["sample_monotonic_ns"] - samples[0]["sample_monotonic_ns"]) / 1e9,
            "max_poll_gap_ms": max(gaps) / 1e6, "metrics": metrics, "events": events,
            "issues": issues, "interpretation":
            "Relative clock evidence only; offset excursion/rate is not proof of a jump, OS cause, simulator alignment or verified actuation."}


def environment() -> dict:
    result = {"platform": platform.platform(), "clock_names": ["monotonic", "realtime", "monotonic_raw", "boottime"]}
    for name, path in (("boot_id", "/proc/sys/kernel/random/boot_id"),
                       ("uptime", "/proc/uptime"),
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
                                 "environment_after": environment(),
                                 "samples": samples, "report": report}, allow_nan=False) + "\n")
    if interrupted:
        raise KeyboardInterrupt
    return report


def reanalyze_probe(source: Path, output: Path) -> dict:
    """Save a separately versioned analysis; never overwrite original evidence."""
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != PROBE_SCHEMA or "samples" not in payload:
        raise ValueError("Expected a saved host-clock-probe/1 document with raw samples")
    report = analyze(payload["samples"])
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "aerowam-host-clock-reanalysis/1",
                                 "source_path": str(source.resolve()),
                                 "source_sha256": hashlib.sha256(raw).hexdigest(),
                                 "source_analysis_profile": payload.get("report", {}).get("analysis_profile", "legacy-adjacent-only"),
                                 "source_interrupted": payload.get("interrupted"),
                                 "report": report}, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--interval-ms", type=float)
    parser.add_argument("--input", type=Path, help="Reanalyze a saved probe into separate output; no clock sampling")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-stable", action="store_true", help="exit 2 for clock inconsistencies/uncertainty")
    args = parser.parse_args()
    if args.input is not None and (args.duration_s is not None or args.interval_ms is not None):
        parser.error("--input cannot be combined with live sampling duration/interval")
    try:
        report = (reanalyze_probe(args.input, args.output) if args.input is not None else
                  run_probe(args.output, duration_s=20. if args.duration_s is None else args.duration_s,
                            interval_ms=50. if args.interval_ms is None else args.interval_ms))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.require_stable and report["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
