"""Self-tests for the public-only warehouse v2 diagnostic baseline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .warehouse_v2_baseline import constant_velocity_predictions, run_baseline_suite, write_report
from .warehouse_v2_task import generate_episode


class WarehouseV2BaselineTests(unittest.TestCase):
    def test_baseline_consumes_public_inputs_and_emits_all_prediction_rows(self):
        record, _ = generate_episode(31, "composed")
        predictions = constant_velocity_predictions(record)
        self.assertEqual(len(predictions), len(record.observations) - 1)
        self.assertEqual(predictions[0]["sample_index"], 0)
        self.assertEqual(predictions[-1]["sample_index"], len(record.observations) - 2)
        self.assertIn("horizons", predictions[0])
        public_wire = json.dumps(predictions, ensure_ascii=True)
        self.assertNotIn("layout_family", public_wire)
        self.assertNotIn("position_enu", public_wire)

    def test_suite_reports_every_split_and_preserves_evidence_boundary(self):
        report = run_baseline_suite(episodes_per_split=1, seed=41)
        self.assertEqual(set(report["splits"]), {"seen", "new_layout", "new_motion", "new_teammate", "composed"})
        self.assertFalse(report["model_score_eligible"])
        self.assertFalse(report["flight_verified"])
        for score in report["splits"].values():
            self.assertGreaterEqual(score["episodes"], 1)
            self.assertIsNotNone(score["prediction_nll"])

    def test_report_writes_reviewable_artifacts(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "baseline"
            write_report(output, episodes_per_split=1, seed=42)
            self.assertTrue((output / "baseline_predictions.jsonl").is_file())
            self.assertTrue((output / "baseline_report.json").is_file())
            self.assertTrue((output / "baseline_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
