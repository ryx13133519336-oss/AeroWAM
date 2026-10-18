"""Hand-computable synthetic scoring fixtures, not measured model performance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .candidate_scoring import (SPEC_SCHEMA, OUTCOME_SCHEMA, EVENTS, canonical, digest,
                                forecast_record, score, validate_spec, _load)
from .decision_interface import Candidate, CandidateStep, CandidatePrediction, DecisionRequest
from .protocol import AGENT_IDS, ModelJointAction, WaypointDelta, assert_truth_free
from .self_test import _observation


def fixture():
    obs = _observation()
    candidates = (Candidate("hold", (CandidateStep(ModelJointAction(), .5),)),
                  Candidate("move", (CandidateStep(ModelJointAction(uav_1=WaypointDelta(dx=.2)), .5),)))
    request = DecisionRequest("synthetic-decision", "search", obs.instruction, (obs,), candidates, .5).to_wire()
    spec = {"schema": SPEC_SCHEMA, "task": "search", "utility_definition": "synthetic_fixture:fixed_0_or_1",
            "repeat_ids": ["r0", "r1"], "utility_tie_epsilon": 0., "top_k": 1,
            "max_collision_probability": None}
    predictions = [CandidatePrediction("hold", .2, {"uav_0": .25, "uav_1": .75}, 0., .2, .1),
                   CandidatePrediction("move", .8, {"uav_0": .75, "uav_1": .25}, 0., .8, .2)]
    forecast = forecast_record(request, spec, predictions, method_id="synthetic_test_not_model", source_kind="synthetic_fixture")
    outcomes = {"schema": OUTCOME_SCHEMA, "decision_id": "synthetic-decision",
                "request_sha256": forecast["request_sha256"], "spec_sha256": digest(spec),
                "source_kind": "synthetic_fixture", "branches": []}
    for name in ("hold", "move"):
        for repeat in spec["repeat_ids"]:
            outcomes["branches"].append({"candidate_id": name, "repeat_id": repeat,
                "paired_start_id": "synthetic-start-" + repeat, "start_matched": True, "status": "complete",
                "terminal_visibility": {"uav_0": name == "move", "uav_1": name == "hold"},
                "events": {"handoff_success_probability": None, "confirmation_probability": name == "move",
                           "collision_probability": False}, "utility": float(name == "move"),
                "evidence_ref": "synthetic://" + name + "/" + repeat})
    return request, spec, forecast, outcomes


def bind(request, spec, forecast, outcomes):
    forecast["request_sha256"] = hashlib.sha256(DecisionRequest.from_wire(request).canonical_bytes()).hexdigest()
    forecast["spec_sha256"] = digest(spec)
    outcomes.update(request_sha256=forecast["request_sha256"], spec_sha256=digest(spec))


def write_synthetic_demo(root):
    """Explicitly synthetic inputs for independent CLI replay, no model/flight."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    request, spec, forecast, outcomes = fixture()
    reverse = deepcopy(forecast)
    reverse["predictions"][0]["utility"] = .9
    reverse["predictions"][1]["utility"] = .1
    incomplete = deepcopy(outcomes)
    incomplete["branches"].pop()
    documents = {"public/request.json": request, "public/spec.json": spec,
                 "public/forecast.json": forecast, "public/forecast_reverse.json": reverse,
                 "evaluator/outcomes.json": outcomes, "evaluator/outcomes_incomplete.json": incomplete,
                 "expected.json": {"source_kind": "synthetic_fixture", "model_inference_performed": False,
                    "normal_regret": 0., "reverse_regret": 1., "utility_mae": .2,
                    "visibility_brier_each_agent": .0625, "confirmation_brier": .04,
                    "collision_brier": .025, "incomplete_status": "ineligible_evidence"}}
    for name, document in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(canonical(document) + b"\n")
    return root


class CandidateScoringTests(unittest.TestCase):
    def test_hand_computed_ranking_regret_utility_and_separate_camera_brier(self):
        result = score(*fixture())
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["outcome_source_kind"], "synthetic_fixture")
        self.assertFalse(result["closed_loop_evaluated"])
        self.assertFalse(result["compositional_transfer_verified"])
        self.assertFalse(result["start_matching_certified_by_scorer"])
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["utility_mae"], .2)
        self.assertAlmostEqual(metrics["utility_rmse"], .2)
        self.assertEqual(metrics["selected_candidate_id"], "move")
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], 0)
        self.assertEqual(metrics["distinguishable_pairs"], 1)
        self.assertEqual(metrics["pairwise_accuracy_half_credit_predicted_ties"], 1)
        self.assertTrue(metrics["top_k_optimal_hit"])
        probs = metrics["probability_scores"]
        for agent in AGENT_IDS:
            self.assertAlmostEqual(probs["terminal_visibility_" + agent]["brier"], .0625)
        self.assertAlmostEqual(probs["confirmation_probability"]["brier"], .04)
        self.assertAlmostEqual(probs["collision_probability"]["brier"], .025)
        self.assertNotIn("handoff_success_probability", probs)

    def test_reverse_ranking_is_failure_not_repaired_by_private_truth(self):
        request, spec, forecast, outcomes = fixture()
        forecast["predictions"][0]["utility"] = .9
        forecast["predictions"][1]["utility"] = .1
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertEqual(metrics["selected_candidate_id"], "hold")
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], 1)
        self.assertFalse(metrics["selected_in_epsilon_optimal_set"])
        self.assertFalse(metrics["top_k_optimal_hit"])
        self.assertEqual(metrics["pairwise_accuracy_half_credit_predicted_ties"], 0)

    def test_brier_is_over_branch_outcomes_not_error_to_average_label(self):
        request, spec, forecast, outcomes = fixture()
        for prediction in forecast["predictions"]:
            prediction["target_visibility_probability"]["uav_0"] = .5
        for row in outcomes["branches"]:
            row["terminal_visibility"]["uav_0"] = row["repeat_id"] == "r1"
        prob = score(request, spec, forecast, outcomes)["metrics"]["probability_scores"]["terminal_visibility_uav_0"]
        self.assertEqual(prob["n_branch_observations"], 4)
        self.assertEqual(prob["predicted_mean"], .5)
        self.assertEqual(prob["observed_rate"], .5)
        self.assertEqual(prob["brier"], .25)  # NOT zero.

    def test_prediction_ties_stable_request_order_and_half_pair_credit(self):
        request, spec, forecast, outcomes = fixture()
        for prediction in forecast["predictions"]:
            prediction["utility"] = .5
        forecast["predictions"].reverse()
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertEqual(metrics["selected_candidate_id"], "hold")
        self.assertEqual(metrics["pairwise_accuracy_half_credit_predicted_ties"], .5)
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], 1)

    def test_frozen_epsilon_handles_true_near_ties_without_division_by_zero(self):
        request, spec, forecast, outcomes = fixture()
        spec["utility_tie_epsilon"] = 1.
        bind(request, spec, forecast, outcomes)
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertEqual(metrics["optimal_candidate_ids"], ["hold", "move"])
        self.assertEqual(metrics["distinguishable_pairs"], 0)
        self.assertIsNone(metrics["pairwise_accuracy_half_credit_predicted_ties"])

    def test_top_k_and_risk_filter_have_explicit_distinct_meanings(self):
        request, spec, forecast, outcomes = fixture()
        spec.update(top_k=2, max_collision_probability=.1)
        bind(request, spec, forecast, outcomes)
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertEqual(metrics["selected_candidate_id"], "hold")
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], 1)
        self.assertEqual(metrics["predicted_top_k"], ["move", "hold"])
        self.assertTrue(metrics["top_k_optimal_hit"])

    def test_abstain_never_gets_zero_regret_or_fabricated_hover_outcome(self):
        request, spec, forecast, outcomes = fixture()
        spec["max_collision_probability"] = 0.
        bind(request, spec, forecast, outcomes)
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertTrue(metrics["abstained"])
        self.assertIsNone(metrics["selected_candidate_id"])
        self.assertIsNone(metrics["utility_regret_vs_all_candidates"])
        self.assertIsNone(metrics["selected_in_epsilon_optimal_set"])

    def test_safety_stops_retained_as_utility_negative_but_probabilities_censored(self):
        request, spec, forecast, outcomes = fixture()
        for row in outcomes["branches"]:
            if row["candidate_id"] == "move":
                row.update(status="safety_stop", utility=-1.)
                row["terminal_visibility"] = dict.fromkeys(AGENT_IDS)
                row["events"] = dict.fromkeys(EVENTS)
                row["events"]["collision_probability"] = True  # Actually observed contact, not inferred from stop.
        result = score(request, spec, forecast, outcomes)
        self.assertEqual(result["status"], "scored")
        metrics = result["metrics"]
        self.assertEqual(metrics["mean_actual_utility"]["move"], -1.)
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], 1.)
        self.assertIsNone(metrics["probability_scores"])
        self.assertEqual(metrics["probability_score_status"], "horizon_censored")

    def test_infrastructure_error_invalidates_whole_matrix_not_model_failure(self):
        for status in ("simulator_error", "evaluator_error"):
            request, spec, forecast, outcomes = fixture()
            row = outcomes["branches"][0]
            row.update(status=status, utility=None, terminal_visibility=dict.fromkeys(AGENT_IDS), events=dict.fromkeys(EVENTS))
            result = score(request, spec, forecast, outcomes)
            self.assertEqual(result["status"], "ineligible_evidence")
            self.assertIsNone(result["metrics"])
            self.assertEqual(result["evidence_issues"][0]["reason"], status)

    def test_missing_or_unmatched_branches_never_score_survivors(self):
        for mutate, reason in ((lambda rows: rows.pop(), "missing_branch"),
                               (lambda rows: rows[0].update(start_matched=False), "start_not_matched"),
                               (lambda rows: rows[0].update(paired_start_id="wrong-start"), "different_paired_starts"),
                               (lambda rows: rows[0].update(utility=None), "utility_unavailable")):
            request, spec, forecast, outcomes = fixture()
            mutate(outcomes["branches"])
            result = score(request, spec, forecast, outcomes)
            self.assertEqual(result["status"], "ineligible_evidence")
            self.assertIsNone(result["metrics"])
            self.assertIn(reason, [issue["reason"] for issue in result["evidence_issues"]])

    def test_duplicate_unknown_or_reused_branch_evidence_rejected(self):
        for mutate in (lambda rows: rows.append(deepcopy(rows[0])),
                       lambda rows: rows[0].update(candidate_id="other"),
                       lambda rows: rows[0].update(repeat_id="extra"),
                       lambda rows: rows[1].update(evidence_ref=rows[0]["evidence_ref"])):
            request, spec, forecast, outcomes = fixture()
            mutate(outcomes["branches"])
            with self.assertRaises(ValueError):
                score(request, spec, forecast, outcomes)

    def test_inapplicable_heads_and_labels_cannot_enter_calibration(self):
        request, spec, forecast, outcomes = fixture()
        forecast["predictions"][0]["handoff_success_probability"] = .1
        with self.assertRaises(ValueError):
            score(request, spec, forecast, outcomes)
        request, spec, forecast, outcomes = fixture()
        outcomes["branches"][0]["events"]["handoff_success_probability"] = False
        with self.assertRaises(ValueError):
            score(request, spec, forecast, outcomes)

    def test_handoff_and_inspection_use_task_specific_heads_without_fixed_roles(self):
        for task in ("handoff", "inspection"):
            request, spec, forecast, outcomes = fixture()
            request["task"] = spec["task"] = task
            for prediction in forecast["predictions"]:
                prediction["confirmation_probability"] = .5 if task == "handoff" else 0.
                prediction["handoff_success_probability"] = .5 if task == "handoff" else 0.
            for row in outcomes["branches"]:
                row["events"]["handoff_success_probability"] = False if task == "handoff" else None
                row["events"]["confirmation_probability"] = True if task == "handoff" else None
            bind(request, spec, forecast, outcomes)
            probs = score(request, spec, forecast, outcomes)["metrics"]["probability_scores"]
            self.assertEqual("handoff_success_probability" in probs, task == "handoff")
            self.assertEqual("confirmation_probability" in probs, task == "handoff")

    def test_safety_censoring_and_complete_evidence_are_not_guessed(self):
        changes = (lambda row: row["terminal_visibility"].update(uav_0=None),
                   lambda row: row["events"].update(confirmation_probability=None),
                   lambda row: row["terminal_visibility"].update(uav_1=1),
                   lambda row: row.update(utility=True),
                   lambda row: row.update(status="simulator_error"),
                   lambda row: row.update(status="safety_stop"),
                   lambda row: row.update(start_matched=1),
                   lambda row: row.update(utility=float("inf")))
        for mutate in changes:
            request, spec, forecast, outcomes = fixture()
            mutate(outcomes["branches"][0])
            with self.assertRaises(ValueError):
                score(request, spec, forecast, outcomes)

    def test_hash_binding_detects_request_spec_or_forecast_mismatch(self):
        changes = (lambda r, s, f, o: f.update(request_sha256="0" * 64),
                   lambda r, s, f, o: o.update(spec_sha256="0" * 64),
                   lambda r, s, f, o: o.update(decision_id="other"),
                   lambda r, s, f, o: s.update(utility_tie_epsilon=.1),
                   lambda r, s, f, o: f.update(decision_id="other"),
                   lambda r, s, f, o: f["predictions"].pop(),
                   lambda r, s, f, o: f["predictions"].append(deepcopy(f["predictions"][0])))
        for mutate in changes:
            data = fixture()
            mutate(*data)
            with self.assertRaises(ValueError):
                score(*data)

    def test_spec_requires_explicit_definitions_repeats_risk_and_tie_rule(self):
        for change in ({"utility_definition": ""}, {"repeat_ids": []}, {"repeat_ids": ["r0", "r0"]},
                       {"repeat_ids": [1]}, {"utility_tie_epsilon": -.1}, {"utility_tie_epsilon": True},
                       {"top_k": True}, {"max_collision_probability": 1.1}, {"top_k": 0}):
            request, spec, forecast, outcomes = fixture()
            spec.update(change)
            with self.assertRaises(ValueError):
                validate_spec(spec)

    def test_public_forecast_never_contains_private_evidence_or_mutates_inputs(self):
        request, spec, forecast, outcomes = fixture()
        before = canonical([request, spec, forecast, outcomes])
        assert_truth_free(request)
        assert_truth_free(forecast)
        result = score(request, spec, forecast, outcomes)
        self.assertEqual(before, canonical([request, spec, forecast, outcomes]))
        self.assertEqual(result["forecast_sha256"], digest(forecast))
        self.assertEqual(result["private_outcomes_sha256"], digest(outcomes))
        self.assertNotIn("private_outcomes", forecast)
        self.assertNotIn("metrics", forecast)

    def test_swapping_uav_ids_preserves_aggregate_utility_and_swaps_visibility_heads(self):
        request, spec, forecast, outcomes = fixture()
        # Make camera metrics distinct so accidentally merging them cannot pass.
        for prediction in forecast["predictions"]:
            prediction["target_visibility_probability"]["uav_0"] = .5
        baseline = score(request, spec, forecast, outcomes)["metrics"]
        def swap(mapping):
            mapping["uav_0"], mapping["uav_1"] = mapping["uav_1"], mapping["uav_0"]
        for obs in request["history"]:
            for key in ("states", "rgb_jpeg_b64", "image_timestamp_ns"):
                swap(obs[key])
            for row in obs["action_history"]:
                swap(row)
        for candidate in request["candidates"]:
            for step in candidate["steps"]:
                swap(step["action"]["waypoints"])
        for prediction in forecast["predictions"]:
            swap(prediction["target_visibility_probability"])
        for row in outcomes["branches"]:
            swap(row["terminal_visibility"])
        bind(request, spec, forecast, outcomes)
        metrics = score(request, spec, forecast, outcomes)["metrics"]
        self.assertEqual(metrics["utility_regret_vs_all_candidates"], baseline["utility_regret_vs_all_candidates"])
        self.assertEqual(metrics["probability_scores"]["terminal_visibility_uav_0"],
                         baseline["probability_scores"]["terminal_visibility_uav_1"])

    def test_rejects_old_fixed_direction_intent_and_numeric_bool(self):
        for mutate in (lambda action: action.update(intent="handoff"),
                       lambda action: action["waypoints"]["uav_0"].__setitem__(0, True)):
            data = fixture()
            mutate(data[0]["candidates"][0]["steps"][0]["action"])
            with self.assertRaises(ValueError):
                score(*data)

    def test_cli_separate_files_no_overwrite_and_insufficient_evidence_exit_two(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = fixture()
            names = ("request", "spec", "forecast", "private-outcomes")
            command = [sys.executable, "-B", "-m", "benchmark.px4_gazebo.model_handoff.candidate_scoring"]
            for name, payload in zip(names, data):
                path = root / (name + ".json")
                path.write_bytes(canonical(payload))
                command.extend(["--" + name, str(path)])
            output = root / "score.json"
            completed = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(_load(output)["status"], "scored")
            saved = output.read_bytes()
            completed = subprocess.run(command + ["--output", str(output)], capture_output=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output.read_bytes(), saved)
            data[3]["branches"].pop()
            (root / "private-outcomes.json").write_bytes(canonical(data[3]))
            completed = subprocess.run(command + ["--output", str(root / "ineligible.json")], capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIsNone(_load(root / "ineligible.json")["metrics"])

    def test_duplicate_json_keys_rejected_by_cli_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.json"
            path.write_text('{"utility":0,"utility":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                _load(path)

    def test_demo_is_explicitly_synthetic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = write_synthetic_demo(Path(folder) / "demo")
            self.assertEqual(_load(root / "evaluator/outcomes.json")["source_kind"], "synthetic_fixture")
            self.assertFalse(_load(root / "expected.json")["model_inference_performed"])
            with self.assertRaises(FileExistsError):
                write_synthetic_demo(root)

    def test_native_action_choice_cannot_masquerade_as_candidate_forecasts(self):
        request, spec, forecast, outcomes = fixture()
        with self.assertRaises(ValueError):
            score(request, spec, {"label": "A", "token_id": 32, "input_tokens": 3063}, outcomes)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-synthetic-demo":
        print(write_synthetic_demo(sys.argv[2]))
    else:
        unittest.main()
