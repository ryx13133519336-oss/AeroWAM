"""Resume serial stair-condition capture with inspection after every episode.

This runner owns one Gazebo collection at a time and never starts PX4. Existing
passed episodes are skipped by their audited condition tuple; failed attempts
remain on disk and are never treated as coverage.
"""

import argparse
import json
from pathlib import Path

from .industrial_targets import parse_component
from .subt_handoff_collect import collect
from .subt_handoff_inspect import contact_sheet, inspect, read_rows


def condition_key(row):
    target = row["target"]
    return (target, bool(row["swapped"]), row["teammate"],
            float(row["message_delay_s"]), bool(row["reverse"]))


def run(package, models, design_path, output_root, *, split="development_training_only", limit=None, existing_episodes=()):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Positive episode limit required")
    design = json.loads(Path(design_path).read_text(encoding="utf8"))
    blocks = [design["development"], design["held_out_route_test"]]
    if split is not None:
        blocks = [block for block in blocks if block["split"] == split]
    planned = [row for block in blocks for row in block["conditions"]]
    if not planned or any(row['message_delay_s'] != .5 for row in planned):
        raise ValueError("Current study fixes communication at 0.5 s; regenerate the capture design")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    done = set()
    for episode in [*output_root.iterdir(), *(Path(path) for path in existing_episodes)]:
        summary_path = episode / "summary.json"
        inspection_path = episode / "inspection.json"
        if not summary_path.is_file() or not inspection_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf8"))
            report = json.loads(inspection_path.read_text(encoding="utf8"))
            condition = summary.get("condition_private")
            if summary.get("dataset_admitted") and report.get("passed") and condition:
                current = inspect(episode)
                if current["files_sha256"] != report.get("files_sha256"):
                    raise ValueError("Episode changed since independent inspection")
                done.add(condition_key({**condition, "target":
                    f"{condition['target']['color']}:{condition['target']['shape']}"}))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    results = []
    attempts = 0
    for row in planned:
        key = condition_key(row)
        if key in done:
            results.append({"condition_id": row["condition_id"], "status": "skipped_passed"})
            continue
        if limit is not None and attempts >= limit:
            break
        stem = row["condition_id"] + ("_descent" if row["reverse"] else "_ascent")
        output = output_root / stem
        attempt = 1
        while output.exists():
            attempt += 1
            output = output_root / f"{stem}_attempt{attempt:03d}"
        attempts += 1
        target = parse_component(row["target"])
        summary = collect(package, models, output, component=target, speed=design["target_speed_m_s"],
                          swapped=row["swapped"], teammate=row["teammate"], reverse=row["reverse"],
                          message_delay_s=row["message_delay_s"], split=row["split"])
        status = "collection_failed"
        if summary.get("dataset_admitted"):
            try:
                report = inspect(output)
                contact_sheet(output, read_rows(output / "public/observations.jsonl"),
                              read_rows(output / "private/annotations.jsonl"),
                              output / "handoff_contact_sheet.png")
                (output / "inspection.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
                status = "collected_and_audited" if report.get("passed") else "audit_failed"
            except Exception as exc:
                status = "audit_failed"
                (output / "batch_audit_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf8")
        results.append({"condition_id": row["condition_id"], "status": status,
                        "output": str(output), "split": row["split"], "reverse": row["reverse"]})
        print(json.dumps(results[-1]), flush=True)
        checkpoint = {"schema": "aerowam-subt-stair-batch/1", "planned": len(planned),
                      "already_passed": sum(condition_key(row) in done for row in planned), "results": results,
                      "px4_started": False, "model_evaluated": False, "flight_admitted": False}
        temporary = output_root / "batch_summary.json.tmp"
        temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf8")
        temporary.replace(output_root / "batch_summary.json")
    result = {"schema": "aerowam-subt-stair-batch/1", "planned": len(planned),
              "already_passed": sum(condition_key(row) in done for row in planned), "results": results,
              "px4_started": False, "model_evaluated": False, "flight_admitted": False}
    (output_root / "batch_summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("development_training_only", "held_out_route_test"),
                        default="development_training_only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--existing-episodes", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(run(args.package, args.models, args.design, args.output_root,
                         split=args.split, limit=args.limit,
                         existing_episodes=args.existing_episodes), indent=2), flush=True)


if __name__ == "__main__":
    main()
