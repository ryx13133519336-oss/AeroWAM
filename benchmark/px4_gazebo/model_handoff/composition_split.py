"""Private, offline audit of declared compositional splits; no data collection.

Metadata consistency is not proof of actual factor interventions, complete
training provenance, independent replication, or compositional transfer.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .adapter_manifest import _load
from .candidate_scoring import canonical, digest
from .language_tasks import TASKS, exact_keys

SCHEMA = "aerowam-composition-split/1-draft"
REPORT_SCHEMA = "aerowam-composition-split-audit/1-draft"
FACTORS = ("task", "target_motion", "observability", "teammate_behavior")
SPLITS = ("train", "validation", "test")
CLAIMS = {"seen_combination", "heldout_combination", "new_structure",
          "heldout_combination_new_structure"}


def _text(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Nonempty, unpadded identifier required")


def _rows(value):
    if not isinstance(value, list):
        raise ValueError("Explicit row list required")
    return value


def _factors(value, behaviors):
    exact_keys(value, set(FACTORS))
    for field in FACTORS:
        _text(value[field])
    if (value["task"] not in TASKS or value["target_motion"] not in {"static", "dynamic"}
            or value["observability"] not in {"finite_fov", "occluded_history"}
            or value["teammate_behavior"] not in behaviors):
        raise ValueError("Unsupported or undeclared factor level")
    return tuple(value[field] for field in FACTORS)


def audit(payload):
    """Reject declared leakage; never infer a split or silently drop branches."""
    payload = json.loads(canonical(payload))  # Detached, finite JSON snapshot.
    exact_keys(payload, {"schema", "design_id", "source_kind", "role_mode",
                         "training_inventory_complete", "teammate_conditions", "composition_axes",
                         "cells", "samples", "exposures"})
    if payload["schema"] != SCHEMA or payload["role_mode"] != "unassigned":
        raise ValueError("Role-unassigned composition schema required")
    _text(payload["design_id"])
    if payload["source_kind"] not in {"synthetic_fixture", "declared_inventory"}:
        raise ValueError("Explicit design/inventory provenance required")
    if type(payload["training_inventory_complete"]) is not bool:
        raise ValueError("Explicit training inventory completeness declaration required")
    axes = payload["composition_axes"]
    if (not isinstance(axes, list) or any(not isinstance(a, str) or a not in FACTORS for a in axes)
            or len(set(axes)) != len(axes) or len(axes) < 2):
        raise ValueError("At least two distinct declared composition axes required")
    def signature(combo):
        return tuple(combo[FACTORS.index(field)] for field in axes)
    behaviors = {}
    for row in _rows(payload["teammate_conditions"]):
        exact_keys(row, {"behavior_id", "kind", "source_ref"})
        _text(row["behavior_id"])
        _text(row["source_ref"])
        if (row["behavior_id"] in behaviors
                or row["kind"] not in {"independent_policy", "command_transform", "unknown"}):
            raise ValueError("Duplicate or unsupported teammate condition")
        behaviors[row["behavior_id"]] = row
    cells, combinations = {}, {}
    for row in _rows(payload["cells"]):
        exact_keys(row, {"cell_id", "split", "claim", "factors", "structure_id"})
        _text(row["cell_id"])
        _text(row["structure_id"])
        if row["cell_id"] in cells or row["split"] not in SPLITS:
            raise ValueError("Duplicate cell or unsupported split")
        if ((row["split"] != "test" and row["claim"] != "development")
                or (row["split"] == "test" and row["claim"] not in CLAIMS)):
            raise ValueError("Split/novelty claim mismatch")
        combinations[row["cell_id"]] = _factors(row["factors"], behaviors)
        cells[row["cell_id"]] = row

    issues = []
    def issue(code, **details):
        issues.append({"code": code, **details})

    if not payload["training_inventory_complete"]:
        issue("training_inventory_incomplete")
    groups = {field: {} for field in ("scene_family_id", "parent_episode_id", "decision_id")}
    contexts, parents, branch_keys, sample_ids, used_cells = {}, {}, set(), set(), set()
    counts = {split: {field: set() for field in groups} for split in SPLITS}
    branches = dict.fromkeys(SPLITS, 0)
    for row in _rows(payload["samples"]):
        exact_keys(row, {"sample_id", "cell_id", "scene_family_id", "parent_episode_id",
                         "decision_id", "candidate_id", "repeat_id"})
        for value in row.values():
            _text(value)
        if row["cell_id"] not in cells or row["sample_id"] in sample_ids:
            raise ValueError("Unknown cell or duplicate sample")
        key = (row["decision_id"], row["candidate_id"], row["repeat_id"])
        if key in branch_keys:
            raise ValueError("Duplicate decision/candidate/repeat branch")
        sample_ids.add(row["sample_id"])
        branch_keys.add(key)
        cell = cells[row["cell_id"]]
        used_cells.add(row["cell_id"])
        split = cell["split"]
        branches[split] += 1
        for field, mapping in groups.items():
            mapping.setdefault(row[field], set()).add(split)
            counts[split][field].add(row[field])
        context = (row["parent_episode_id"], row["scene_family_id"], row["cell_id"])
        contexts.setdefault(row["decision_id"], set()).add(context)
        parents.setdefault(row["parent_episode_id"], set()).add(row["scene_family_id"])
    for field, mapping in groups.items():
        for name, splits in sorted(mapping.items()):
            if len(splits) > 1:
                issue("group_crosses_splits", field=field, group_id=name, splits=sorted(splits))
    for name, values in sorted(contexts.items()):
        if len(values) > 1:
            issue("decision_context_conflict", decision_id=name)
    for name, values in sorted(parents.items()):
        if len(values) > 1:
            issue("parent_scene_family_conflict", parent_episode_id=name)
    for cell_id in sorted(set(cells) - used_cells):
        issue("cell_without_inventory", cell_id=cell_id)
    for split in SPLITS:
        if not branches[split]:
            issue("empty_split", split=split)

    # Only populated cells contribute support. Pretraining/finetuning exposures
    # count as training exposure even if excluded from the new split manifest.
    train, tuned, train_structures, tuned_structures, used_behaviors = set(), set(), set(), set(), set()
    for cell_id in sorted(used_cells):
        row, combo = cells[cell_id], combinations[cell_id]
        used_behaviors.add(combo[-1])
        if row["split"] == "train":
            train.add(combo)
            train_structures.add(row["structure_id"])
        elif row["split"] == "validation":
            tuned.add(combo)
            tuned_structures.add(row["structure_id"])
    exposure_ids = set()
    for row in _rows(payload["exposures"]):
        exact_keys(row, {"exposure_id", "usage", "factors", "structure_id", "source_ref"})
        for field in ("exposure_id", "structure_id", "source_ref"):
            _text(row[field])
        if row["exposure_id"] in exposure_ids or row["usage"] not in {"pretraining", "finetuning", "validation_tuning"}:
            raise ValueError("Duplicate or unsupported prior exposure")
        exposure_ids.add(row["exposure_id"])
        combo = _factors(row["factors"], behaviors)
        (tuned if row["usage"] == "validation_tuning" else train).add(combo)
        used_behaviors.add(combo[-1])
        (tuned_structures if row["usage"] == "validation_tuning" else train_structures).add(row["structure_id"])
    levels = {field: sorted({combo[i] for combo in train}) for i, field in enumerate(FACTORS)}
    train_signatures, tuned_signatures = {signature(c) for c in train}, {signature(c) for c in tuned}
    for name in sorted(used_behaviors):
        if behaviors[name]["kind"] != "independent_policy":
            issue("teammate_proxy_is_not_policy_variation", behavior_id=name, kind=behaviors[name]["kind"])
    for cell_id in sorted(used_cells):
        row, combo = cells[cell_id], combinations[cell_id]
        if row["split"] != "test":
            continue
        novel_combo = "heldout_combination" in row["claim"]
        novel_structure = "new_structure" in row["claim"]
        if novel_combo:
            if signature(combo) in train_signatures:
                issue("heldout_combination_exposed_in_training", cell_id=cell_id)
            if signature(combo) in tuned_signatures:
                issue("heldout_combination_exposed_in_tuning", cell_id=cell_id)
            for i, field in enumerate(FACTORS):
                if combo[i] not in levels[field]:
                    issue("heldout_factor_not_seen_in_training", cell_id=cell_id, factor=field, level=combo[i])
        elif combo not in train:
            issue("claimed_seen_combination_not_in_training", cell_id=cell_id)
        structure_seen = row["structure_id"] in (train_structures | tuned_structures if novel_structure else train_structures)
        if novel_structure == structure_seen:
            issue("structure_novelty_claim_mismatch", cell_id=cell_id, structure_id=row["structure_id"])

    represented = set(combinations[cell_id] for cell_id in used_cells)
    grid_size = math.prod(len({c[i] for c in represented}) for i in range(len(FACTORS)))
    return {"schema": REPORT_SCHEMA, "design_id": payload["design_id"],
            "inventory_sha256": digest(payload), "source_kind": payload["source_kind"],
            "status": "rejected" if issues else "declared_split_consistent", "issues": issues,
            "composition_axes": axes,
            "training_factor_levels_including_prior_exposure": levels,
            "counts_by_split": {split: {"scene_families": len(counts[split]["scene_family_id"]),
                "parent_episodes": len(counts[split]["parent_episode_id"]),
                "decisions": len(counts[split]["decision_id"]), "candidate_repeat_branches": branches[split]}
                for split in SPLITS},
            "represented_combinations": len(represented), "full_factorial_grid_size": grid_size,
            "independent_replicate_count": None, "factorial_effects_identifiability_verified": False,
            "training_provenance_verified": False, "factor_interventions_verified": False,
            "branch_completeness_verified": False, "start_matching_verified": False,
            "role_free_execution_verified": False, "formal_split_frozen": False,
            "compositional_transfer_verified": False, "flight_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    result = audit(_load(raw))
    result["inventory_file_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "issues": len(result["issues"])}))
    raise SystemExit(0 if not result["issues"] else 2)


if __name__ == "__main__":
    main()
