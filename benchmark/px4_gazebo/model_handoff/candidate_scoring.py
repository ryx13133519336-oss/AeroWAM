"""Independent private scoring of candidate forecasts, never a policy input.

Consumes externally evaluated paired branches. Does not execute candidates,
create truth, certify start matching, or turn an on-policy log into branches.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from .decision_interface import CandidatePrediction, DecisionRequest, select, validate_predictions
from .language_tasks import exact_keys, finite_number
from .protocol import AGENT_IDS

SPEC_SCHEMA = "aerowam-candidate-scoring-spec/1-draft"
FORECAST_SCHEMA = "aerowam-candidate-forecast-record/1-draft"
OUTCOME_SCHEMA = "aerowam-private-candidate-outcomes/1-draft"
SCORE_SCHEMA = "aerowam-candidate-score/1-draft"
EVENTS = ("handoff_success_probability", "confirmation_probability", "collision_probability")
STATUSES = {"complete", "safety_stop", "simulator_error", "evaluator_error"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty text required")


def validate_spec(spec):
    exact_keys(spec, {"schema", "task", "utility_definition", "repeat_ids", "utility_tie_epsilon",
                      "top_k", "max_collision_probability"})
    if spec["schema"] != SPEC_SCHEMA or spec["task"] not in {"search", "handoff", "inspection"}:
        raise ValueError("Scoring spec schema/task mismatch")
    _text(spec["utility_definition"])
    repeats = spec["repeat_ids"]
    if not isinstance(repeats, list) or not repeats:
        raise ValueError("Explicit planned repeats required")
    for repeat in repeats:
        _text(repeat)
    if len(set(repeats)) != len(repeats):
        raise ValueError("Duplicate planned repeat")
    finite_number(spec["utility_tie_epsilon"], "utility epsilon")
    if spec["utility_tie_epsilon"] < 0 or type(spec["top_k"]) is not int or spec["top_k"] < 1:
        raise ValueError("Nonnegative epsilon and positive integer top_k required")
    risk = spec["max_collision_probability"]
    if risk is not None:
        finite_number(risk, "risk threshold")
        if not 0 <= risk <= 1:
            raise ValueError("Risk threshold outside [0,1]")


def _request(payload, spec):
    validate_spec(spec)
    request = DecisionRequest.from_wire(payload)
    if request.task != spec["task"] or spec["top_k"] > len(request.candidates):
        raise ValueError("Scoring task/top_k differs from candidate request")
    # Reject boolean numeric coercions and legacy fixed-direction intents here.
    for candidate in payload["candidates"]:
        for step in candidate["steps"]:
            if step["action"]["intent"] != "none":
                raise ValueError("This role-unassigned scorer rejects legacy fixed-direction intents")
            finite_number(step["duration_s"], "candidate duration")
            for values in step["action"]["waypoints"].values():
                for value in values:
                    finite_number(value, "candidate waypoint")
    return request


def forecast_record(request_payload, spec, predictions, *, method_id, source_kind):
    """Harness wrapper before private outcomes are read; no model call or repairs."""
    request = _request(request_payload, spec)
    _text(method_id)
    if source_kind not in {"synthetic_fixture", "model_output"}:
        raise ValueError("Explicit forecast provenance required")
    predictions = tuple(predictions)
    validate_predictions(request, predictions)
    result = {"schema": FORECAST_SCHEMA, "decision_id": request.decision_id,
              "request_sha256": hashlib.sha256(request.canonical_bytes()).hexdigest(),
              "spec_sha256": digest(spec), "method_id": method_id, "source_kind": source_kind,
              "predictions": [row.to_wire() for row in predictions]}
    # Canonical detached copy so caller mutation cannot change a sealed value.
    return json.loads(canonical(result))


def _events(task):
    if task == "handoff":
        return EVENTS
    if task == "search":
        return EVENTS[1:]
    return EVENTS[2:]


def _outcomes(bundle, request, spec, request_hash):
    exact_keys(bundle, {"schema", "decision_id", "request_sha256", "spec_sha256", "source_kind", "branches"})
    if (bundle["schema"] != OUTCOME_SCHEMA or bundle["decision_id"] != request.decision_id
            or bundle["request_sha256"] != request_hash or bundle["spec_sha256"] != digest(spec)
            or bundle["source_kind"] not in {"synthetic_fixture", "recorded_branches"}):
        raise ValueError("Private outcome provenance/request/spec mismatch")
    if not isinstance(bundle["branches"], list):
        raise ValueError("Explicit private branch rows required")
    candidates = {c.candidate_id for c in request.candidates}
    expected = set(itertools.product(candidates, spec["repeat_ids"]))
    rows, issues, starts, references = {}, [], {}, set()
    for row in bundle["branches"]:
        exact_keys(row, {"candidate_id", "repeat_id", "paired_start_id", "start_matched", "status",
                         "terminal_visibility", "events", "utility", "evidence_ref"})
        key = row["candidate_id"], row["repeat_id"]
        if key not in expected or key in rows:
            raise ValueError("Unknown or duplicate candidate/repeat branch")
        _text(row["paired_start_id"])
        _text(row["evidence_ref"])
        if row["evidence_ref"] in references:
            raise ValueError("One branch evidence reference cannot stand in for several branches")
        references.add(row["evidence_ref"])
        if type(row["start_matched"]) is not bool or row["status"] not in STATUSES:
            raise ValueError("Invalid branch status/start-match flag")
        exact_keys(row["terminal_visibility"], set(AGENT_IDS))
        exact_keys(row["events"], set(EVENTS))
        labels = [*row["terminal_visibility"].values(), *row["events"].values()]
        if any(value is not None and type(value) is not bool for value in labels):
            raise ValueError("Private labels must be bool or explicitly unavailable")
        if any(row["events"][event] is not None for event in EVENTS if event not in _events(request.task)):
            raise ValueError("Inapplicable task events must be unavailable, not false evidence")
        if row["utility"] is not None:
            finite_number(row["utility"], "independent branch utility")
        if row["status"] == "complete":
            if any(row["terminal_visibility"][a] is None for a in AGENT_IDS) or any(
                    row["events"][event] is None for event in _events(request.task)):
                raise ValueError("Complete window requires all applicable labels")
        elif row["status"] == "safety_stop":
            if any(v is not None for v in row["terminal_visibility"].values()):
                raise ValueError("Early safety stop cannot invent terminal visibility")
            if any(v is False for v in row["events"].values()):
                raise ValueError("Unobserved events in truncated windows are unknown, not negative")
        elif row["utility"] is not None or any(v is not None for v in labels):
            raise ValueError("Infrastructure errors cannot be assigned negative outcomes")
        rows[key] = row
        starts.setdefault(row["repeat_id"], set()).add(row["paired_start_id"])
        if not row["start_matched"]:
            issues.append({"candidate_id": key[0], "repeat_id": key[1], "reason": "start_not_matched"})
        if row["status"] in {"simulator_error", "evaluator_error"}:
            issues.append({"candidate_id": key[0], "repeat_id": key[1], "reason": row["status"]})
        elif row["utility"] is None:
            issues.append({"candidate_id": key[0], "repeat_id": key[1], "reason": "utility_unavailable"})
    for candidate, repeat in sorted(expected - set(rows)):
        issues.append({"candidate_id": candidate, "repeat_id": repeat, "reason": "missing_branch"})
    for repeat, identifiers in starts.items():
        if len(identifiers) != 1:
            issues.append({"repeat_id": repeat, "reason": "different_paired_starts"})
    return rows, issues


def _probability_score(probabilities, labels):
    n = len(labels)
    return {"n_branch_observations": n,
            "brier": math.fsum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n,
            "predicted_mean": math.fsum(probabilities) / n,
            "observed_rate": math.fsum(labels) / n}


def score(request_payload, spec, forecast, private_outcomes):
    """One decision, all planned candidates/repeats. No cross-frame averaging."""
    # Validate detached snapshots of mutable nested dictionaries.
    request_payload, spec, forecast, private_outcomes = json.loads(
        canonical([request_payload, spec, forecast, private_outcomes]))
    request = _request(request_payload, spec)
    request_hash = hashlib.sha256(request.canonical_bytes()).hexdigest()
    exact_keys(forecast, {"schema", "decision_id", "request_sha256", "spec_sha256", "method_id",
                          "source_kind", "predictions"})
    if (forecast["schema"] != FORECAST_SCHEMA or forecast["decision_id"] != request.decision_id
            or forecast["request_sha256"] != request_hash or forecast["spec_sha256"] != digest(spec)):
        raise ValueError("Forecast bound to different request/spec")
    _text(forecast["method_id"])
    if forecast["source_kind"] not in {"synthetic_fixture", "model_output"}:
        raise ValueError("Invalid forecast provenance")
    predictions = tuple(CandidatePrediction.from_wire(row) for row in forecast["predictions"])
    validate_predictions(request, predictions)
    for prediction in predictions:
        if any(getattr(prediction, event) != 0 for event in EVENTS if event not in _events(request.task)):
            raise ValueError("Inapplicable probability heads must be declared zero, never scored")
    rows, issues = _outcomes(private_outcomes, request, spec, request_hash)
    result = {"schema": SCORE_SCHEMA, "decision_id": request.decision_id,
              "request_sha256": request_hash, "spec_sha256": digest(spec),
              "forecast_sha256": digest(forecast), "private_outcomes_sha256": digest(private_outcomes),
              "method_id": forecast["method_id"], "forecast_source_kind": forecast["source_kind"],
              "outcome_source_kind": private_outcomes["source_kind"],
              "scope": "single_decision_offline_not_closed_loop_or_transfer",
              "closed_loop_evaluated": False, "compositional_transfer_verified": False,
              "start_matching_certified_by_scorer": False,
              "status": "ineligible_evidence" if issues else "scored", "evidence_issues": issues,
              "metrics": None}
    if issues:
        return result  # Never drop failed branches/candidates and score survivors.
    by_id = {prediction.candidate_id: prediction for prediction in predictions}
    identifiers = [candidate.candidate_id for candidate in request.candidates]
    actual = {name: math.fsum(rows[name, repeat]["utility"] for repeat in spec["repeat_ids"]) /
                    len(spec["repeat_ids"]) for name in identifiers}
    errors = [by_id[name].utility - actual[name] for name in identifiers]
    best = max(actual.values())
    optimal = [name for name in identifiers if best - actual[name] <= spec["utility_tie_epsilon"]]
    chosen = select(request, predictions, max_collision_probability=spec["max_collision_probability"])
    ranked = sorted(identifiers, key=lambda name: by_id[name].utility, reverse=True)
    pair_credits = []
    for a, b in itertools.combinations(identifiers, 2):
        if abs(actual[a] - actual[b]) <= spec["utility_tie_epsilon"]:
            continue
        difference = by_id[a].utility - by_id[b].utility
        pair_credits.append(.5 if difference == 0 else float((difference > 0) == (actual[a] > actual[b])))
    metrics = {"mean_actual_utility": actual, "utility_mae": math.fsum(abs(e) for e in errors) / len(errors),
               "utility_rmse": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
               "selected_candidate_id": chosen, "abstained": chosen is None,
               "optimal_candidate_ids": optimal,
               "selected_in_epsilon_optimal_set": None if chosen is None else chosen in optimal,
               "utility_regret_vs_all_candidates": None if chosen is None else best - actual[chosen],
               "predicted_top_k": ranked[:spec["top_k"]],
               "top_k_optimal_hit": bool(set(optimal) & set(ranked[:spec["top_k"]])),
               "distinguishable_pairs": len(pair_credits),
               "pairwise_accuracy_half_credit_predicted_ties": math.fsum(pair_credits) / len(pair_credits) if pair_credits else None,
               "probability_scores": None, "probability_score_status": "horizon_censored"}
    # Preserve safety-stop utilities in ranking, but don't score a false horizon
    # negative or selectively include only observed-positive censored labels.
    if all(row["status"] == "complete" for row in rows.values()):
        keys = list(rows)
        probabilities = {}
        for agent in AGENT_IDS:
            probabilities["terminal_visibility_" + agent] = _probability_score(
                [by_id[name].target_visibility_probability[agent] for name, repeat in keys],
                [rows[key]["terminal_visibility"][agent] for key in keys])
        for event in _events(request.task):
            probabilities[event] = _probability_score(
                [getattr(by_id[name], event) for name, repeat in keys], [rows[key]["events"][event] for key in keys])
        metrics.update(probability_scores=probabilities, probability_score_status="complete_windows")
    result["metrics"] = metrics
    canonical(result)  # Reject arithmetic overflow instead of emitting NaN/Inf.
    return result


def _load(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate JSON key")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "spec", "forecast", "private-outcomes", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = score(_load(args.request), _load(args.spec), _load(args.forecast), _load(args.private_outcomes))
    # Exclusive create: original forecasts, labels and old scores are untouched.
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "decision_id": result["decision_id"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "scored" else 2)


if __name__ == "__main__":
    main()
