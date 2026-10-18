"""Copy frozen validation records, fixing only the identity metric denominator."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..evaluator import replay_metric_differences
from ..live_handoff_runner import validate_episode_record
from .fixtures import read_manifest


def repair_record(record: dict) -> tuple[dict, dict]:
    """Never change trajectory, events, termination or success."""
    import copy
    validate_episode_record(record)
    differences = replay_metric_differences(record)
    if set(differences) - {"identity_error_rate"}:
        raise ValueError(f"Unsupported replay differences: {differences}")
    result = copy.deepcopy(record)
    if differences:
        result["outcome"]["metrics"]["identity_error_rate"] = differences["identity_error_rate"][1]
    validate_episode_record(result)
    if replay_metric_differences(result):
        raise ValueError("Repaired record still differs from replay")
    return result, differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Snapshot output already exists: {args.output_dir}")
    manifest = Path(__file__).resolve().parent / "manifests/handoff_wall_blocking_v1.jsonl"
    fixtures = {f.fixture_id: f for f in read_manifest(manifest) if f.split == "validation"}
    staged = {}
    provenance = []
    counts = {}
    for root in args.inputs:
        paths = sorted(p for p in root.rglob("*.jsonl") if ".attempts" not in p.parts)
        if not paths:
            raise ValueError(f"No final records at {root}")
        for path in paths:
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) != 1:
                raise ValueError(f"Expected one record: {path}")
            raw = json.loads(lines[0])
            model = raw["task_spec"]["model"]
            fixture_id = raw["task_spec"]["fixture_id"]
            fixture = fixtures.get(fixture_id)
            if fixture is None or raw["seed"] != fixture.seed or raw["task_spec"]["fixture_split"] != "validation":
                raise ValueError(f"Fixture identity mismatch: {path}")
            if raw["outcome"]["termination_reason"] == "simulator_error":
                raise ValueError(f"Simulator failure cannot enter snapshot: {path}")
            group = (model["method"], int(model["training_seed"]))
            if group[0] not in {"joint_bc", "dual_vln_hier", "dual_wam_mpc"} or group[1] < 0:
                raise ValueError(f"Unsupported method/seed: {group}")
            if group in counts and counts[group]["checkpoint_id"] != model["checkpoint_id"]:
                raise ValueError(f"Mixed checkpoint IDs in {group}")
            summary = counts.setdefault(group, {"checkpoint_id": model["checkpoint_id"], "fixtures": set(), "success": 0, "terminations": Counter()})
            if fixture_id in summary["fixtures"]:
                raise ValueError(f"Duplicate fixture in {group}: {fixture_id}")
            summary["fixtures"].add(fixture_id)
            summary["success"] += int(raw["outcome"]["success"])
            summary["terminations"][raw["outcome"]["termination_reason"]] += 1
            fixed, differences = repair_record(raw)
            relative = Path(group[0]) / f"seed_{group[1]}" / "validation" / f"{fixture_id}.jsonl"
            encoded = json.dumps(fixed, ensure_ascii=False) + "\n"
            staged[relative] = encoded
            provenance.append({"source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "snapshot": str(relative), "snapshot_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "metric_differences": differences})
    for group, summary in counts.items():
        if summary["fixtures"] != set(fixtures):
            raise ValueError(f"Incomplete coverage for {group}")
    # Validate everything before creating the independent snapshot.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for relative, encoded in staged.items():
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    (args.output_dir / "metric_repair_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    report = ["# Frozen validation baseline results", "", "Single training seed; validation results only. No language-understanding or general world-model claim.", "", "| Method | Seed | Success | Terminations |", "|---|---:|---:|---|"]
    for (method, seed), summary in sorted(counts.items()):
        report.append(f"| {method} | {seed} | {summary['success']}/{len(fixtures)} | {dict(summary['terminations'])} |")
    report.extend(["", "Only identity_error_rate was normalized to serialized identity events. Original files, task success and collision results are preserved."])
    (args.output_dir / "baseline.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Snapshot complete: {len(staged)} records; corrected={sum(bool(p['metric_differences']) for p in provenance)}; {args.output_dir / 'baseline.md'}")


if __name__ == "__main__":
    main()
