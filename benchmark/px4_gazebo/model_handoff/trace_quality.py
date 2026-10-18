"""Offline timing diagnostics; receive ages never imply exposure alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .execution_trace import BOUNDED_TRACE_SCHEMA, TIMED_SCHEMAS, validate_trace
from .protocol import AGENT_IDS

QUALITY_PROFILE = "aerowam-trace-receive-quality/2-draft"
QUALITY_REPORT_FILE = "trace_receive_quality.json"
# Provisional offline gates, independent of task success and flight safety.
LIMITS_MS = {"publication_gap": 100.0, "odometry_receive_age": 250.0,
             "image_receive_age": 250.0, "unix_clock_jump": 10.0}


def clock_pair_diagnostics(stream: list[dict]) -> tuple[list[float], list[float]]:
    """Minimum offset change outside brackets, plus read uncertainty widths.

    UNIX minus monotonic lies in [UNIX-after, UNIX-before] at the UNIX read.
    Disjoint intervals indicate inconsistent offsets; a broad interval only
    indicates uncertain timing and must not prove a wall-clock jump.
    """
    widths = [(sample["clock_read_end_monotonic_ns"] - sample["received_monotonic_ns"]) / 1e6
              for sample in stream]
    changes = []
    for a, b in zip(stream, stream[1:]):
        a_low = a["received_unix_ns"] - a["clock_read_end_monotonic_ns"]
        a_high = a["received_unix_ns"] - a["received_monotonic_ns"]
        b_low = b["received_unix_ns"] - b["clock_read_end_monotonic_ns"]
        b_high = b["received_unix_ns"] - b["received_monotonic_ns"]
        changes.append(max(0, b_low - a_high, a_low - b_high) / 1e6)
    return changes, widths


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "max_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {"count": len(values), "max_ms": ordered[-1],
            "p95_ms": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))]}


def quality_report(header: Mapping, rows: list[dict]) -> dict:
    """Validate structure, then flag timing issues without excluding outcomes.

    Gazebo clock receive pairs and TimesyncStatus are evidence, not an exact
    simulator/host or camera/odometry mapping. DDS may already adjust stamps;
    no estimated_offset is added/subtracted from received odometry here.
    """
    validate_trace(header, rows)
    issues: list[str] = []
    metrics = {}
    for agent in AGENT_IDS:
        own = [row for row in rows if row["agent_id"] == agent]
        gaps = [row["duration_s"] * 1000 for row in own]
        ages = [(row[f"{end}_monotonic_ns"] - row[f"{end}_odometry"]["received_monotonic_ns"]) / 1e6
                for row in own for end in ("start", "end")]
        metrics[agent] = {"publication_gap": _stats(gaps), "odometry_receive_age": _stats(ages)}
        for name, values in (("publication_gap", gaps), ("odometry_receive_age", ages)):
            if max(values) > LIMITS_MS[name]:
                issues.append(f"{agent}:{name}_above_draft_limit")
        previous = None
        for row in own:
            for endpoint in ("start", "end"):
                stamp = row[f"{endpoint}_odometry"]["timestamp_sample_us"]
                if previous is not None and stamp < previous:
                    issues.append(f"{agent}:odometry_sample_clock_backward")
                    break
                previous = stamp
        command_jumps = [abs((b["command_timestamp_us"] - a["command_timestamp_us"]) / 1000
                             - (b["start_monotonic_ns"] - a["start_monotonic_ns"]) / 1e6)
                         for a, b in zip(own, own[1:])]
        metrics[agent]["command_unix_monotonic_delta"] = _stats(command_jumps)
        if command_jumps and max(command_jumps) > LIMITS_MS["unix_clock_jump"]:
            issues.append(f"{agent}:command_unix_clock_jump")

    timed = header["schema"] in TIMED_SCHEMAS
    bounded = header["schema"] == BOUNDED_TRACE_SCHEMA
    evidence_counts = {"sim_clock": 0, "timesync": {agent: 0 for agent in AGENT_IDS}}
    if not timed:
        issues.append("legacy_trace_missing_timing_evidence")
    else:
        anchors = header["sample_anchors"]
        for agent in AGENT_IDS:
            ages = [(sample["capture_monotonic_ns"] - sample["timing"]["image_received_monotonic_ns"][agent]) / 1e6
                    for sample in anchors]
            metrics[agent]["image_receive_age"] = _stats(ages)
            if max(ages) > LIMITS_MS["image_receive_age"]:
                issues.append(f"{agent}:image_receive_age_above_draft_limit")
            stamps = [sample["image_timestamp_ns"][agent] for sample in anchors]
            if any(b < a for a, b in zip(stamps, stamps[1:])):
                issues.append(f"{agent}:image_clock_backward")
            if any(b == a for a, b in zip(stamps, stamps[1:])):
                issues.append(f"{agent}:reused_image_stamp")
        evidence = header["timing_evidence"]
        clock = evidence["sim_clock"]
        evidence_counts["sim_clock"] = len(clock)
        if len(clock) < 2:
            issues.append("insufficient_sim_clock_evidence")
        elif clock[-1]["stamp_ns"] <= clock[0]["stamp_ns"]:
            issues.append("sim_clock_did_not_advance")
        if any(b["stamp_ns"] < a["stamp_ns"] for a, b in zip(clock, clock[1:])):
            issues.append("sim_clock_backward")
        for agent in AGENT_IDS:
            sync = evidence["timesync"][agent]
            evidence_counts["timesync"][agent] = len(sync)
            if len(sync) < 2 or any(sample["source_protocol"] != 2 for sample in sync):
                issues.append(f"{agent}:insufficient_dds_timesync_evidence")
        # Compare each producer's paired host readings in receive order.
        # The reads are sequential, not atomic: an inconsistency may mean a
        # wall-clock adjustment OR preemption between reads. Neither is safe
        # evidence of a precise clock mapping.
        host_streams = [clock, anchors_to_clock(anchors), *evidence["timesync"].values()]
        jumps = []
        widths = []
        for stream in host_streams:
            if bounded:
                changes, uncertainty = clock_pair_diagnostics(stream)
                jumps.extend(changes)
                widths.extend(uncertainty)
            else:
                jumps.extend(abs((b["received_unix_ns"] - a["received_unix_ns"])
                                 - (b["received_monotonic_ns"] - a["received_monotonic_ns"])) / 1e6
                             for a, b in zip(stream, stream[1:]))
        metrics["host_unix_monotonic_delta"] = _stats(jumps)
        if jumps and max(jumps) > LIMITS_MS["unix_clock_jump"]:
            issues.append("host_clock_pair_inconsistency")
        if bounded:
            metrics["host_clock_read_uncertainty"] = _stats(widths)
            if widths and max(widths) > LIMITS_MS["unix_clock_jump"]:
                issues.append("host_clock_read_uncertainty_above_draft_limit")
        else:
            issues.append("missing_host_clock_read_bounds")
    return {"profile": QUALITY_PROFILE, "episode_id": header["episode_id"],
            "limits_ms": LIMITS_MS, "metrics": metrics, "evidence_counts": evidence_counts,
            "receive_quality_pass": not issues, "issues": sorted(set(issues)),
            "clock_alignment": "unverified", "precise_transition_supervision_ready": False,
            "reason": "Receive freshness and clock evidence do not establish sensor-generation alignment or actuation."}


def anchors_to_clock(anchors: list[dict]) -> list[dict]:
    return [{"received_monotonic_ns": sample["capture_monotonic_ns"],
             "received_unix_ns": sample["timing"]["capture_unix_ns"],
             **({"clock_read_end_monotonic_ns": sample["timing"]["clock_read_end_monotonic_ns"]}
                if "clock_read_end_monotonic_ns" in sample["timing"] else {})} for sample in anchors]


def report_episode(episode_dir: Path, *, persist: bool = False) -> dict:
    """Offline report kept beside evaluator artifacts, never a policy label.

    Does not change task outcome or delete/repair any recorded observation.
    Persist only newly collected attempts; existing files are never overwritten.
    """
    entries = [json.loads(line) for line in (episode_dir / "policy" / "published_controls.jsonl")
               .read_text(encoding="utf-8").splitlines() if line]
    if not entries:
        raise ValueError("Empty trace")
    report = quality_report(entries[0], entries[1:])
    if report["episode_id"] != episode_dir.name:
        raise ValueError("Quality report/episode identity mismatch")
    if persist:
        with (episode_dir / "evaluator" / QUALITY_REPORT_FILE).open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--require-receive-quality", action="store_true",
                        help="exit 2 for draft receive-quality violations; never rewrites task results")
    args = parser.parse_args()
    entries = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line]
    if not entries:
        raise ValueError("Empty trace")
    report = quality_report(entries[0], entries[1:])
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.require_receive_quality and not report["receive_quality_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
