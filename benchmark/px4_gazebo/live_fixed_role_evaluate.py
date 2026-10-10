"""Offline aggregation for live FixedRole PX4-Gazebo diagnostic episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence


TASK_ID = "live_dual_control_observation_diagnostic"
RATE_METRICS = ("control_success", "pre_position_success", "collision_rate")
SUCCESS_METRICS = (
    "preposition_error_m", "minimum_separation_m", "flight_time_s",
    "camera_frames_uav_0", "camera_frames_uav_1", "communication_bytes",
)
REQUIRED_METRICS = RATE_METRICS + SUCCESS_METRICS


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in inputs:
        if source.is_dir():
            paths.extend(sorted(source.rglob("*.jsonl")))
        else:
            paths.append(source)
    return list(dict.fromkeys(path.resolve() for path in paths))


def load_live_records(
    inputs: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    issues: list[dict[str, Any]] = []
    duplicate_count = 0
    total_lines = 0
    paths = discover_inputs(inputs)
    for path in paths:
        if not path.exists():
            issues.append({"source": str(path), "line": 0, "reason": "path does not exist"})
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            issues.append({"source": str(path), "line": 0, "reason": f"read error: {exc}"})
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"source": str(path), "line": line_number, "reason": f"invalid JSON: {exc}"})
                continue
            if not isinstance(record, dict) or record.get("task_id") != TASK_ID:
                issues.append({"source": str(path), "line": line_number, "reason": "not a live diagnostic episode"})
                continue
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                issues.append({"source": str(path), "line": line_number, "reason": "missing episode_id"})
                continue
            if episode_id in seen_episode_ids:
                duplicate_count += 1
                continue
            outcome = record.get("outcome")
            metrics = outcome.get("metrics") if isinstance(outcome, dict) else None
            missing = [
                name for name in REQUIRED_METRICS
                if not isinstance(metrics, dict) or _number(metrics.get(name)) is None
            ]
            if (
                not isinstance(outcome, dict)
                or not isinstance(outcome.get("success"), bool)
                or not isinstance(outcome.get("termination_reason"), str)
                or missing
            ):
                issues.append({
                    "source": str(path), "line": line_number,
                    "reason": "invalid outcome/metrics; missing or non-finite: " + ", ".join(missing),
                })
                continue
            seen_episode_ids.add(episode_id)
            records.append(record)
    metadata = {
        "input_files": [str(path) for path in paths],
        "nonempty_lines": total_lines,
        "accepted_records": len(records),
        "duplicate_records": duplicate_count,
        "skipped_or_invalid_records": len(issues),
        "issues": issues,
    }
    return records, metadata


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean": None, "median": None, "std_population": None}
    return {
        "count": len(data), "mean": statistics.fmean(data),
        "median": statistics.median(data), "std_population": statistics.pstdev(data),
    }


def summarize_records(
    records: Sequence[dict[str, Any]], metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successful = [record for record in records if record["outcome"]["success"]]
    failed = [record for record in records if not record["outcome"]["success"]]
    reasons = Counter(record["outcome"]["termination_reason"] for record in records)
    rates: dict[str, float | None] = {
        "success_rate": (
            statistics.fmean(float(record["outcome"]["success"]) for record in records)
            if records else None
        )
    }
    for name in RATE_METRICS:
        rates[name] = (
            statistics.fmean(float(record["outcome"]["metrics"][name]) for record in records)
            if records else None
        )
    success_stats = {
        name: _describe(record["outcome"]["metrics"][name] for record in successful)
        for name in SUCCESS_METRICS
    }
    success_stats["minimum_separation_m"]["minimum"] = (
        min(record["outcome"]["metrics"]["minimum_separation_m"] for record in successful)
        if successful else None
    )
    for agent in ("uav_0", "uav_1"):
        frames_key = f"camera_frames_{agent}"
        success_stats[f"camera_fps_proxy_{agent}"] = _describe(
            record["outcome"]["metrics"][frames_key]
            / record["outcome"]["metrics"]["flight_time_s"]
            for record in successful
            if record["outcome"]["metrics"]["flight_time_s"] > 0.0
        )
    return {
        "task_id": TASK_ID,
        "input": metadata or {},
        "groups": {
            "all": {"count": len(records), **rates},
            "successful": {
                "count": len(successful),
                "episode_ids": [record["episode_id"] for record in successful],
                "metrics": success_stats,
            },
            "failed": {
                "count": len(failed),
                "episodes": [
                    {
                        "episode_id": record["episode_id"], "seed": record.get("seed"),
                        "termination_reason": record["outcome"]["termination_reason"],
                    }
                    for record in failed
                ],
            },
            "termination_reason_counts": dict(sorted(reasons.items())),
        },
    }


def _format_value(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def render_markdown(summary: dict[str, Any]) -> str:
    groups = summary["groups"]
    all_group = groups["all"]
    successful = groups["successful"]
    lines = [
        "# Live FixedRole Summary", "",
        f"- Accepted episodes: {all_group['count']}",
        f"- Successful / failed: {successful['count']} / {groups['failed']['count']}",
        f"- Success rate: {_format_value(all_group['success_rate'])}",
        f"- Control success rate: {_format_value(all_group['control_success'])}",
        f"- Pre-position success rate: {_format_value(all_group['pre_position_success'])}",
        f"- Collision rate: {_format_value(all_group['collision_rate'])}",
        f"- Duplicate records ignored: {summary['input'].get('duplicate_records', 0)}",
        f"- Invalid/skipped records: {summary['input'].get('skipped_or_invalid_records', 0)}",
        "", "## Successful Episodes", "",
        "| Metric | Count | Mean | Median | Std (population) | Minimum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in successful["metrics"].items():
        lines.append("| " + " | ".join([
            name, _format_value(stats.get("count")), _format_value(stats.get("mean")),
            _format_value(stats.get("median")), _format_value(stats.get("std_population")),
            _format_value(stats.get("minimum")),
        ]) + " |")
    lines.extend(["", "## Termination Reasons", ""])
    for reason, count in groups["termination_reason_counts"].items():
        lines.append(f"- {reason}: {count}")
    if groups["failed"]["episodes"]:
        lines.extend(["", "## Failed Episodes", ""])
        for episode in groups["failed"]["episodes"]:
            lines.append(f"- seed={episode['seed']} episode={episode['episode_id']}: {episode['termination_reason']}")
    issues = summary["input"].get("issues", [])
    if issues:
        lines.extend(["", "## Invalid or Skipped Lines", ""])
        for issue in issues:
            lines.append(f"- {issue['source']}:{issue['line']}: {issue['reason']}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate live FixedRole diagnostic JSONL files")
    parser.add_argument("inputs", type=Path, nargs="+", help="JSONL file(s) or directories")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    records, metadata = load_live_records(args.inputs)
    summary = summarize_records(records, metadata)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_text, encoding="utf-8")
    print(json_text if args.format == "json" else markdown_text, end="")


if __name__ == "__main__":
    main()
