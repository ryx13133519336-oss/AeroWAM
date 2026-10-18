"""Synthetic split inventories, never real training or transfer measurements."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .candidate_scoring import canonical
from .composition_split import SCHEMA, audit


def fixture():
    result = {"schema": SCHEMA, "design_id": "synthetic_split_not_frozen",
              "source_kind": "synthetic_fixture", "role_mode": "unassigned",
              "training_inventory_complete": True, "composition_axes": ["task", "target_motion"],
              "teammate_conditions": [{"behavior_id": b, "kind": "independent_policy",
                                       "source_ref": "synthetic://policy/" + b} for b in ("normal", "cautious")],
              "cells": [], "samples": [], "exposures": []}
    def add(name, split, claim, task, motion, obs, behavior, structure="known"):
        result["cells"].append({"cell_id": name, "split": split, "claim": claim,
            "structure_id": structure, "factors": {"task": task, "target_motion": motion,
                "observability": obs, "teammate_behavior": behavior}})
        for candidate in ("hold", "move"):
            for repeat in ("r0", "r1"):
                result["samples"].append({"sample_id": name + "-" + candidate + "-" + repeat,
                    "cell_id": name, "scene_family_id": "family-" + name,
                    "parent_episode_id": "parent-" + name, "decision_id": "decision-" + name,
                    "candidate_id": candidate, "repeat_id": repeat})
    add("train_handoff", "train", "development", "handoff", "static", "finite_fov", "normal")
    add("train_search", "train", "development", "search", "dynamic", "occluded_history", "cautious")
    add("train_inspection", "train", "development", "inspection", "static", "occluded_history", "normal")
    add("val", "validation", "development", "handoff", "static", "finite_fov", "normal")
    add("seen", "test", "seen_combination", "search", "dynamic", "occluded_history", "cautious")
    add("combo", "test", "heldout_combination", "handoff", "dynamic", "occluded_history", "cautious")
    add("structure", "test", "new_structure", "search", "dynamic", "occluded_history", "cautious", "new")
    add("both", "test", "heldout_combination_new_structure", "handoff", "dynamic",
        "occluded_history", "cautious", "new")
    return result


def exposure(inventory, cell="combo", usage="pretraining"):
    row = next(r for r in inventory["cells"] if r["cell_id"] == cell)
    return {"exposure_id": "old_model_data", "usage": usage, "factors": deepcopy(row["factors"]),
            "structure_id": row["structure_id"], "source_ref": "synthetic://old-training"}


def write_synthetic_demo(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    valid = fixture()
    leaked = deepcopy(valid)
    leaked["exposures"].append(exposure(leaked))
    for name, data in (("inventory.json", valid), ("inventory_training_leak.json", leaked)):
        with (root / name).open("xb") as stream:
            stream.write(canonical(data) + b"\n")
    return root


class SplitTests(unittest.TestCase):
    def codes(self, inventory):
        return {r["code"] for r in audit(inventory)["issues"]}

    def test_valid_four_test_conditions_and_no_capability_claim(self):
        report = audit(fixture())
        self.assertEqual(report["status"], "declared_split_consistent")
        self.assertEqual(report["issues"], [])
        self.assertIsNone(report["independent_replicate_count"])
        for key, value in report.items():
            if key.endswith("verified") or key in {"formal_split_frozen", "flight_ready"}:
                self.assertIs(value, False, key)

    def test_count_branches_separately_from_parents_and_scenes(self):
        counts = audit(fixture())["counts_by_split"]["test"]
        self.assertEqual(counts, {"scene_families": 4, "parent_episodes": 4,
                                  "decisions": 4, "candidate_repeat_branches": 16})

    def test_candidate_repeat_cannot_cross_split_at_any_group_level(self):
        for field in ("scene_family_id", "parent_episode_id", "decision_id"):
            with self.subTest(field=field):
                data = fixture()
                sample = next(r for r in data["samples"] if r["cell_id"] == "combo")
                sample[field] = data["samples"][0][field]
                sample["candidate_id"] = "other"  # Keep the branch key unique.
                self.assertIn("group_crosses_splits", self.codes(data))

    def test_parent_and_decision_context_must_be_consistent_even_within_split(self):
        data = fixture()
        data["samples"][1]["scene_family_id"] = "different-family"
        self.assertIn("parent_scene_family_conflict", self.codes(data))
        self.assertIn("decision_context_conflict", self.codes(data))

    def test_old_dynamic_handoff_pretraining_leaks_even_if_other_axes_differ(self):
        data = fixture()
        row = exposure(data)
        row["factors"].update(observability="finite_fov", teammate_behavior="normal")
        data["exposures"].append(row)
        self.assertIn("heldout_combination_exposed_in_training", self.codes(data))

    def test_finetuning_also_counts_as_exposure(self):
        data = fixture()
        data["exposures"].append(exposure(data, usage="finetuning"))
        self.assertIn("heldout_combination_exposed_in_training", self.codes(data))

    def test_current_training_cell_cannot_contain_holdout_signature(self):
        data = fixture()
        data["cells"][0]["factors"]["target_motion"] = "dynamic"
        self.assertIn("heldout_combination_exposed_in_training", self.codes(data))

    def test_real_inventory_label_does_not_certify_source_or_execution(self):
        data = fixture()
        data["source_kind"] = "declared_inventory"
        report = audit(data)
        self.assertEqual(report["status"], "declared_split_consistent")
        self.assertFalse(report["training_provenance_verified"])
        self.assertFalse(report["factor_interventions_verified"])

    def test_validation_tuning_cannot_see_test_combination(self):
        data = fixture()
        data["exposures"].append(exposure(data, usage="validation_tuning"))
        self.assertIn("heldout_combination_exposed_in_tuning", self.codes(data))

    def test_validation_cells_are_tuning_exposure(self):
        data = fixture()
        data["cells"][3]["factors"] = deepcopy(data["cells"][5]["factors"])
        self.assertIn("heldout_combination_exposed_in_tuning", self.codes(data))

    def test_unseen_atomic_level_is_not_recombination(self):
        data = fixture()
        data["teammate_conditions"].append({"behavior_id": "unseen", "kind": "independent_policy",
                                            "source_ref": "synthetic://unseen"})
        data["cells"][5]["factors"]["teammate_behavior"] = "unseen"
        self.assertIn("heldout_factor_not_seen_in_training", self.codes(data))

    def test_empty_declared_cell_cannot_supply_training_support(self):
        data = fixture()
        data["samples"] = [r for r in data["samples"] if r["cell_id"] != "train_search"]
        codes = self.codes(data)
        self.assertIn("cell_without_inventory", codes)
        self.assertIn("heldout_factor_not_seen_in_training", codes)

    def test_new_structure_cannot_be_exposed_in_training_or_tuning(self):
        for usage in ("pretraining", "validation_tuning"):
            with self.subTest(usage=usage):
                data = fixture()
                data["exposures"].append(exposure(data, cell="structure", usage=usage))
                self.assertIn("structure_novelty_claim_mismatch", self.codes(data))

    def test_seen_structure_requires_training_not_only_validation_exposure(self):
        data = fixture()
        data["cells"][3]["structure_id"] = "validation_only"
        data["cells"][4]["structure_id"] = "validation_only"
        self.assertIn("structure_novelty_claim_mismatch", self.codes(data))

    def test_seen_claim_cannot_hide_an_unseen_full_combination(self):
        data = fixture()
        data["cells"][4]["factors"]["teammate_behavior"] = "normal"
        self.assertIn("claimed_seen_combination_not_in_training", self.codes(data))

    def test_delay_and_scale_proxy_cannot_certify_teammate_policy_variation(self):
        for kind in ("command_transform", "unknown"):
            data = fixture()
            data["teammate_conditions"][1]["kind"] = kind
            self.assertIn("teammate_proxy_is_not_policy_variation", self.codes(data))

    def test_incomplete_training_inventory_refuses_clean_holdout_claim(self):
        data = fixture()
        data["training_inventory_complete"] = False
        self.assertIn("training_inventory_incomplete", self.codes(data))

    def test_no_partition_can_be_empty(self):
        data = fixture()
        data["samples"] = [r for r in data["samples"] if r["cell_id"] != "val"]
        self.assertIn("empty_split", self.codes(data))

    def test_exact_fields_and_no_preassigned_role(self):
        for field, value in (("role_mode", "fixed"), ("source_kind", "measured_transfer"),
                             ("roles", {"uav_0": "scout"}), ("training_inventory_complete", 1)):
            data = fixture()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit(data)

    def test_duplicate_branch_even_with_different_sample_id_is_rejected(self):
        data = fixture()
        row = deepcopy(data["samples"][0])
        row["sample_id"] = "duplicate"
        data["samples"].append(row)
        with self.assertRaisesRegex(ValueError, "Duplicate decision"):
            audit(data)

    def test_composition_axes_must_be_explicit_distinct_supported_and_multiple(self):
        for axes in ([], ["task"], ["task", "task"], ["task", "color"], ["task", []]):
            data = fixture()
            data["composition_axes"] = axes
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                audit(data)

    def test_input_is_unchanged_and_report_has_detached_axes(self):
        data = fixture()
        before = deepcopy(data)
        report = audit(data)
        data["composition_axes"].reverse()
        self.assertEqual(report["composition_axes"], before["composition_axes"])
        data["composition_axes"].reverse()
        self.assertEqual(data, before)

    def test_cli_success_rejection_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, output = root / "inventory.json", root / "audit.json"
            request.write_bytes(canonical(fixture()))
            command = [sys.executable, "-B", "-m", "benchmark.px4_gazebo.model_handoff.composition_split",
                       "--inventory", str(request), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = output.read_bytes()
            self.assertEqual(json.loads(before)["status"], "declared_split_consistent")
            result = subprocess.run(command, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            bad = fixture()
            bad["exposures"].append(exposure(bad))
            request.write_bytes(canonical(bad))
            command[-1] = str(root / "rejected.json")
            self.assertEqual(subprocess.run(command, capture_output=True, timeout=15).returncode, 2)

    def test_cli_rejects_duplicate_json_keys_before_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "bad.json", root / "output.json"
            source.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-m",
                "benchmark.px4_gazebo.model_handoff.composition_split", "--inventory", str(source),
                "--output", str(output)], capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_synthetic_demo_is_explicit_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_synthetic_demo(Path(directory) / "demo")
            valid = json.loads((root / "inventory.json").read_bytes())
            self.assertEqual(valid["source_kind"], "synthetic_fixture")
            self.assertEqual(audit(valid)["status"], "declared_split_consistent")
            leaked = json.loads((root / "inventory_training_leak.json").read_bytes())
            self.assertIn("heldout_combination_exposed_in_training", self.codes(leaked))
            with self.assertRaises(FileExistsError):
                write_synthetic_demo(root)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-synthetic-demo":
        print(write_synthetic_demo(sys.argv[2]))
    else:
        unittest.main()
