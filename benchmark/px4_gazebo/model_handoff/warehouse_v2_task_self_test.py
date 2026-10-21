"""Self-test for the first warehouse v2 cooperation/prediction task."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .core.episode_schema import episode_from_dict
from .core.model_io import ModelInput
from .warehouse_v2_task import (
    AGENTS, HORIZONS_S, LAYOUT_FAMILIES, MOTION_FAMILIES, SPLIT_MATRIX,
    TEAMMATE_FAMILIES, TASK, SCENE, export_dataset,
    generate_episode, model_inputs, score_predictions,
)


class WarehouseV2TaskTests(unittest.TestCase):
    def test_protocol_declares_required_latent_families(self):
        self.assertGreaterEqual(len(LAYOUT_FAMILIES), 6)
        self.assertGreaterEqual(len(MOTION_FAMILIES), 5)
        self.assertGreaterEqual(len(TEAMMATE_FAMILIES), 5)

    def test_episode_has_public_private_boundary_and_core_roundtrip(self):
        record, truth = generate_episode(7, "composed")
        restored = episode_from_dict(record.to_dict())
        self.assertEqual(restored.canonical_bytes(), record.canonical_bytes())
        self.assertEqual(truth["layout_family"], "l_turn")
        # The core record intentionally carries private supervision for the
        # trusted evaluator.  The public boundary is the model-input wire.
        public_wire = json.dumps([item.to_wire() for item in model_inputs(record)], ensure_ascii=True)
        for forbidden in ("layout_family", "motion_family", "teammate_family", "branch", "position_enu"):
            self.assertNotIn(forbidden, public_wire)
        self.assertTrue(record.private_audit.kinematic_proxy)
        self.assertFalse(record.private_audit.model_score_eligible)

    def test_public_model_inputs_are_valid_and_future_targets_are_private(self):
        record, _ = generate_episode(8, "seen")
        inputs = model_inputs(record)
        self.assertEqual(len(inputs), len(record.observations))
        self.assertTrue(all(isinstance(item, ModelInput) for item in inputs))
        self.assertEqual(len(record.future_observation_supervision), len(record.observations) - 1)
        public = json.dumps([item.to_wire() for item in inputs], ensure_ascii=True)
        self.assertNotIn("position_enu", public)
        self.assertNotIn("motion_family", public)

    def test_baseline_prediction_scores_and_unsafe_request_is_penalized(self):
        record, truth = generate_episode(9, "new_motion")
        unsafe_index = next(
            index for index, row in enumerate(truth["rows"][:-1])
            if row["visible_by_agent"][AGENTS[0]] and not row["visible_by_agent"][AGENTS[1]]
        )
        predictions = []
        for index in range(len(record.observations) - 1):
            predictions.append({
                "sample_index": index,
                "horizons": {str(h): {"mean_y": 0.0, "std_y": 1.0,
                                       "visible_probability": {agent: 0.5 for agent in AGENTS}}
                              for h in HORIZONS_S},
                "action": {"event": "request" if index == unsafe_index else "wait"},
            })
        score = score_predictions(record, truth, predictions)
        self.assertEqual(score["prediction_samples"], len(record.observations) - 1)
        self.assertGreater(score["prediction_nll"], 0.0)
        self.assertGreater(score["unsafe_handoff_rate"], 0.0)
        self.assertFalse(score["model_score_eligible"])

    def test_split_matrix_is_explicit_and_export_is_reproducible(self):
        self.assertEqual(set(SPLIT_MATRIX), {"seen", "new_layout", "new_motion", "new_teammate", "composed"})
        with TemporaryDirectory() as folder:
            manifest = export_dataset(Path(folder) / "dataset", episodes_per_split=1, seed=33)
            self.assertEqual(manifest["episodes"], 5)
            self.assertFalse(manifest["core_schema_modified"])
            self.assertTrue((Path(folder) / "dataset" / "private_truth").is_dir())
            public_episode_lines = (Path(folder) / "dataset" / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(public_episode_lines), 5)
            self.assertNotIn("position_enu", "\n".join(public_episode_lines))
            self.assertNotIn("future_observation_supervision", "\n".join(public_episode_lines))
            self.assertEqual(manifest["schema"], "aerowam-warehouse-v2-dataset/2")
            self.assertEqual(TASK.task_id, "warehouse_coop_prediction")
            self.assertEqual(SCENE.scene_id, "aws_warehouse_handoff_v2")


if __name__ == "__main__":
    unittest.main()
