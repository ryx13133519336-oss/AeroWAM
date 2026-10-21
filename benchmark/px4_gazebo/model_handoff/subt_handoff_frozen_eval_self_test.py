"""Frozen offline evaluation regression tests; no synthetic benchmark evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .subt_handoff_design import design
from .subt_handoff_frozen_eval import (
    ARTIFACTS, BOUNDARIES, Persistence, audit, complete_test_paths,
    event_diagnostics, verified_checkpoint,
)
from .subt_handoff_learn import load_episode, sha256

PREFIX = "benchmark.px4_gazebo.model_handoff."


class FixedPolicy:
    def predict(self, inputs):
        return np.asarray(["none", "found", "none", "handoff_request"])


class FrozenEvaluationTests(unittest.TestCase):
    def test_none_accuracy_does_not_hide_false_events(self):
        episode = {"features": np.zeros((2, 419)), "events": [
            {"uav_0": "none", "uav_1": "none"},
            {"uav_0": "found", "uav_1": "handoff_request"}]}
        report = event_diagnostics(FixedPolicy(), [episode])
        self.assertEqual(report["false_event_reports_on_none"], 1)
        self.assertEqual(report["missed_events_predicted_none"], 1)
        self.assertEqual(report["per_event"]["found"]["precision"], 0.)
        self.assertEqual(report["per_event"]["found"]["recall"], 0.)
        self.assertEqual(report["always_none_accuracy"], .5)

    def test_persistence_uses_current_images_and_pose_only(self):
        features = np.arange(838).reshape(2, 419)
        np.testing.assert_array_equal(Persistence().predict(features), features[:, :398])

    def test_test_matrix_requires_all_eight_and_reaudits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, condition in enumerate(design()["held_out_route_test"]["conditions"]):
                path = root / str(i)
                path.mkdir()
                color, shape = condition["target"].split(":")
                report = {"passed": True, "split": "held_out_route_test",
                    "route_id": "stair_descent_v1", "message_delay_s": .5,
                    "condition": {**condition, "target": {"color": color, "shape": shape}},
                    "files_sha256": {"data": str(i)}}
                (path / "inspection.json").write_text(json.dumps(report), encoding="utf8")
            def fresh(path):
                return json.loads((path / "inspection.json").read_text(encoding="utf8"))
            with patch(PREFIX + "subt_handoff_frozen_eval.inspect", side_effect=fresh) as inspector:
                self.assertEqual(len(complete_test_paths(root)), 8)
                self.assertEqual(inspector.call_count, 8)
            (root / "0/batch_audit_error.txt").write_text("rejected on capture platform")
            with patch(PREFIX + "subt_handoff_frozen_eval.inspect", side_effect=fresh):
                with self.assertRaisesRegex(ValueError, "incomplete: 7/8"):
                    complete_test_paths(root)

    def test_known_failed_audit_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batch_audit_error.txt").write_text("failed")
            with self.assertRaisesRegex(ValueError, "cannot be bypassed"):
                load_episode(root, "held_out_route_test")

    def test_changed_checkpoint_refused_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ARTIFACTS:
                (root / name).write_text("fixture")
            hashes = {name: sha256(root / name) for name in ARTIFACTS}
            (root / "result.json").write_text(json.dumps({"artifacts_sha256": hashes}))
            (root / ARTIFACTS[0]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verified_checkpoint(root)

    def test_audit_rejects_score_boundary_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation.json").write_text(json.dumps(BOUNDARIES))
            (root / "result.json").write_text(json.dumps({**BOUNDARIES, "success": True}))
            with self.assertRaisesRegex(ValueError, "boundary changed"):
                audit(root)

    def test_audit_recomputes_and_rejects_metric_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            fingerprints = {"model": "frozen"}
            plan = {**BOUNDARIES, "checkpoint_dir": str(root),
                "artifacts_sha256": fingerprints, "test_root": str(root),
                "extra_test_episodes": [], "sources": {str(source): {"data": "frozen"}}}
            computed = {**BOUNDARIES, "episodes": 8, "diagnostic": 0.3}
            (root / "evaluation.json").write_text(json.dumps(plan))
            (root / "result.json").write_text(json.dumps(computed))
            with patch(PREFIX + "subt_handoff_frozen_eval.verified_checkpoint",
                       return_value=(root, fingerprints, object(), object())), \
                 patch(PREFIX + "subt_handoff_frozen_eval.complete_test_paths", return_value=[source]), \
                 patch(PREFIX + "subt_handoff_frozen_eval.load_episode",
                       return_value={"root": source, "report": {"files_sha256": {"data": "frozen"}}}), \
                 patch(PREFIX + "subt_handoff_frozen_eval.calculate", return_value=computed):
                self.assertTrue(audit(root)["passed"])
                (root / "result.json").write_text(json.dumps({**computed, "diagnostic": 0.8}))
                with self.assertRaisesRegex(ValueError, "recomputation differs"):
                    audit(root)


if __name__ == "__main__":
    unittest.main()
