"""Offline summary for live Handoff PX4/Gazebo episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .contract import HandoffTask
from .evaluator import episode_metrics, replay_metric_differences


TASK_ID = HandoffTask().task_id


def discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in inputs:
        if source.is_dir():
            paths.extend(sorted(source.rglob("*.jsonl")))
        else:
            paths.append(source)
    return list(dict.fromkeys(path.resolve() for path in paths))


def load_records(inputs: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    issues: list[dict[str, Any]] = []
    duplicates = 0
    lines_seen = 0
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
            lines_seen += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"source": str(path), "line": line_number, "reason": f"invalid JSON: {exc}"})
                continue
            if not isinstance(record, dict) or record.get("task_id") != TASK_ID:
                issues.append({"source": str(path), "line": line_number, "reason": "not a live Handoff episode"})
                continue
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                issues.append({"source": str(path), "line": line_number, "reason": "missing episode_id"})
                continue
            if episode_id in seen:
                duplicates += 1
                continue
            outcome = record.get("outcome")
            if not isinstance(outcome, Mapping) or not isinstance(outcome.get("success"), bool):
                issues.append({"source": str(path), "line": line_number, "reason": "missing boolean outcome.success"})
                continue
            differences = replay_metric_differences(record)
            if differences:
                issues.append({"source": str(path), "line": line_number, "reason": "recorded metrics disagree with replay", "differences": differences})
                continue
            seen.add(episode_id)
            records.append(record)
    return records, {
        "input_files": [str(path) for path in paths],
        "nonempty_lines": lines_seen,
        "accepted_records": len(records),
        "duplicate_records": duplicates,
        "invalid_records": len(issues),
        "issues": issues,
    }


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean": None, "median": None, "std_population": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "std_population": statistics.pstdev(data),
    }


def summarize(records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    rows = [episode_metrics(record) for record in records]
    successful = [row for row, record in zip(rows, records) if bool(record.get("outcome", {}).get("success"))]
    metric_names = sorted(rows[0].keys()) if rows else []
    return {
        "schema_version": "0.2",
        "task_id": TASK_ID,
        "accepted_episodes": len(records),
        "successful_episodes": sum(bool(record.get("outcome", {}).get("success")) for record in records),
        "failed_episodes": sum(not bool(record.get("outcome", {}).get("success")) for record in records),
        "success_rate": (sum(bool(record.get("outcome", {}).get("success")) for record in records) / len(records)) if records else 0.0,
        "metadata": dict(metadata),
        "all_metrics": {name: _describe(row[name] for row in rows) for name in metric_names},
        "successful_metrics": {name: _describe(row[name] for row in successful) for name in metric_names},
        "episodes": [
            {"episode_id": record.get("episode_id"), "seed": record.get("seed"), "success": bool(record.get("outcome", {}).get("success")), "termination_reason": record.get("outcome", {}).get("termination_reason"), "metrics": row}
            for record, row in zip(records, rows)
        ],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Live Handoff Summary",
        "",
        f"- Accepted episodes: {summary['accepted_episodes']}",
        f"- Successful / failed: {summary['successful_episodes']} / {summary['failed_episodes']}",
        f"- Success rate: {summary['success_rate']:.4f}",
        f"- Invalid/skipped records: {summary['metadata']['invalid_records']}",
        f"- Duplicate records ignored: {summary['metadata']['duplicate_records']}",
        "",
        "## All Episodes",
        "",
        "| Metric | Count | Mean | Median | Std (population) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in summary["all_metrics"].items():
        lines.append(f"| {name} | {stats['count']} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['std_population']:.4f} |" if stats["count"] else f"| {name} | 0 | n/a | n/a | n/a |")
    lines.extend(["", "## Termination Reasons", ""])
    reasons: dict[str, int] = {}
    for episode in summary["episodes"]:
        reason = str(episode["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live Handoff episode JSONL files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    records, metadata = load_records(args.inputs)
    summary = summarize(records, metadata)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(rendered, end="")


if __name__ == "__main__":
    main()
