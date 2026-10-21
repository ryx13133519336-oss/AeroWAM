"""Independent aggregation of eligible SubT handoff evaluation reports.

The leaderboard consumes completed model-run reports.  It never runs Gazebo,
PX4 or a model, and it refuses teacher/kinematic-proxy evidence as formal
model results.  Diagnostic proxy runs remain visible in the audit output but
are never ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REPORT_SCHEMA = "aerowam-subt-handoff-evaluation/1"
LEADERBOARD_SCHEMA = "aerowam-subt-handoff-leaderboard/1"
STATUSES = {"complete", "model_failure", "safety_stop", "simulator_error", "evaluator_error"}
METHOD_KINDS = {"external_model", "repository_model", "reference_rule", "teacher"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "benchmark_version", "method_id", "method_kind", "model_manifest_sha256",
        "task_id", "scene_id", "scene_family_id", "split", "episode_id", "condition",
        "status", "success", "metrics", "evidence", "limitations",
    }
    if not isinstance(report, Mapping) or set(report) != expected or report["schema"] != REPORT_SCHEMA:
        raise ValueError("evaluation report schema or fields are invalid")
    for key in ("benchmark_version", "method_id", "model_manifest_sha256", "task_id", "scene_id", "scene_family_id", "split", "episode_id"):
        _text(report[key], key)
    if report["method_kind"] not in METHOD_KINDS:
        raise ValueError("unknown method_kind")
    if len(report["model_manifest_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in report["model_manifest_sha256"]):
        raise ValueError("model_manifest_sha256 must be lowercase SHA256")
    condition = report["condition"]
    condition_keys = {"condition_id", "target", "swapped", "teammate", "message_delay_s", "route_id", "repeat_id"}
    if not isinstance(condition, Mapping) or set(condition) != condition_keys:
        raise ValueError("condition fields are invalid")
    for key in ("condition_id", "target", "teammate", "route_id", "repeat_id"):
        _text(condition[key], f"condition.{key}")
    _bool(condition["swapped"], "condition.swapped")
    if _finite(condition["message_delay_s"], "condition.message_delay_s") != 0.5:
        raise ValueError("formal SubT leaderboard requires fixed 0.5 s communication delay")
    if report["status"] not in STATUSES:
        raise ValueError("unknown evaluation status")
    if report["success"] is not None:
        _bool(report["success"], "success")
    if report["status"] == "complete" and report["success"] is None:
        raise ValueError("complete reports require a boolean success")
    if report["status"] != "complete" and report["success"] is not None:
        raise ValueError("non-complete reports must leave success null")
    if not isinstance(report["metrics"], Mapping):
        raise ValueError("metrics must be an object")
    for key, value in report["metrics"].items():
        _finite(value, f"metrics.{key}")
    evidence = report["evidence"]
    evidence_keys = {
        "closed_loop", "public_input_only", "independent_audit", "flight_admitted",
        "privileged_teacher", "kinematic_proxy", "action_trace_verified", "role_assignment",
        "external_model", "audit_sha256", "trace_sha256", "public_request_sha256", "private_outcome_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_keys:
        raise ValueError("evidence fields are invalid")
    for key in ("closed_loop", "public_input_only", "independent_audit", "flight_admitted", "privileged_teacher", "kinematic_proxy", "action_trace_verified", "external_model"):
        _bool(evidence[key], f"evidence.{key}")
    if evidence["role_assignment"] is not None:
        raise ValueError("role_assignment must be null")
    for key in ("audit_sha256", "trace_sha256", "public_request_sha256", "private_outcome_sha256"):
        digest = evidence[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"evidence.{key} must be lowercase SHA256")
    if not isinstance(report["limitations"], list) or any(not isinstance(item, str) or not item.strip() for item in report["limitations"]):
        raise ValueError("limitations must be a non-empty-text list")
    # A formal row must come from a real external/repository model acting in
    # the scene.  Teacher and kinematic-proxy rows are retained as diagnostics.
    eligible = (
        report["status"] == "complete" and report["success"] is not None
        and report["method_kind"] in {"external_model", "repository_model"}
        and evidence["closed_loop"] and evidence["public_input_only"]
        and evidence["independent_audit"] and evidence["flight_admitted"]
        and evidence["action_trace_verified"] and evidence["external_model"]
        and not evidence["privileged_teacher"] and not evidence["kinematic_proxy"]
    )
    return {**dict(report), "_model_score_eligible": eligible}


def _load_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw)
        validated = validate_report(report)
        validated["_report_sha256"] = _sha256(raw)
        validated["_report_path"] = str(Path(path).resolve())
        reports.append(validated)
    if not reports:
        raise ValueError("at least one evaluation report is required")
    return reports


def aggregate(reports: Iterable[Mapping[str, Any]], *, planned_condition_keys: set[tuple[str, str]] | None = None, minimum_eligible: int = 1) -> dict[str, Any]:
    if type(minimum_eligible) is not int or minimum_eligible < 1:
        raise ValueError("minimum_eligible must be a positive integer")
    validated = [validate_report(row) for row in reports]
    seen = set()
    for row in validated:
        condition = row["condition"]
        key = (row["method_id"], row["split"], condition["condition_id"], condition["repeat_id"])
        if key in seen:
            raise ValueError(f"duplicate method/condition/repeat report: {key}")
        seen.add(key)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        grouped[(row["method_id"], row["split"])].append(row)
    rows = []
    for (method_id, split), items in sorted(grouped.items()):
        eligible = [row for row in items if row["_model_score_eligible"]]
        successes = sum(row["success"] is True for row in eligible)
        condition_keys = {(row["condition"]["condition_id"], row["condition"]["repeat_id"]) for row in eligible}
        coverage_ok = planned_condition_keys is None or planned_condition_keys <= condition_keys
        ranked = len(eligible) >= minimum_eligible and coverage_ok
        latency = [row["metrics"]["latency_ms"] for row in eligible if "latency_ms" in row["metrics"]]
        rows.append({
            "method_id": method_id, "split": split, "ranked": ranked,
            "n_reports": len(items), "n_eligible": len(eligible),
            "n_ineligible": len(items) - len(eligible),
            "n_success": successes,
            "success_rate": successes / len(eligible) if eligible else None,
            "mean_latency_ms": sum(latency) / len(latency) if latency else None,
            "coverage_ok": coverage_ok,
            "coverage": sorted({f"{a}:{b}" for a, b in condition_keys}),
            "ineligible_reasons": sorted({
                "status_or_success" if row["status"] != "complete" or row["success"] is None else
                "teacher_or_kinematic_proxy" if row["evidence"]["privileged_teacher"] or row["evidence"]["kinematic_proxy"] else
                "not_flight_admitted" if not row["evidence"]["flight_admitted"] else
                "not_closed_loop_or_trace_unverified"
                for row in items if not row["_model_score_eligible"]
            }),
            "report_sha256": sorted(row["_report_sha256"] for row in items if "_report_sha256" in row),
        })
    ranked_rows = sorted((row for row in rows if row["ranked"]), key=lambda row: (-row["success_rate"], row["mean_latency_ms"] if row["mean_latency_ms"] is not None else float("inf"), row["method_id"]))
    for index, row in enumerate(ranked_rows, 1):
        row["rank"] = index
    for row in rows:
        row.setdefault("rank", None)
    return {
        "schema": LEADERBOARD_SCHEMA,
        "formal_score": bool(ranked_rows),
        "ranking": ranked_rows,
        "all_methods": rows,
        "rules": {
            "unit": "eligible_complete_episode",
            "success_rate": "n_success / n_eligible",
            "communication_delay_s": 0.5,
            "teacher_and_kinematic_proxy_excluded": True,
            "infrastructure_failures_excluded_from_denominator": True,
            "minimum_eligible": minimum_eligible,
            "planned_coverage_required": planned_condition_keys is not None,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-eligible", type=int, default=1)
    args = parser.parse_args(argv)
    result = aggregate(_load_reports(args.reports), minimum_eligible=args.minimum_eligible)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "formal_score": result["formal_score"], "ranked": len(result["ranking"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
