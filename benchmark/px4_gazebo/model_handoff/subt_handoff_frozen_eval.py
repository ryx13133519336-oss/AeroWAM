"""Evaluate frozen exploratory checkpoints on all eight audited descent cells.

Recorded teacher messages are inputs; this is offline diagnostic evaluation,
not a closed-loop handoff score or action-conditioned world-model evaluation.
The separate audit command reconstructs every metric without refitting.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .subt_handoff_batch import condition_key
from .subt_handoff_design import design as capture_design
from .subt_handoff_inspect import inspect
from .subt_handoff_learn import (
    AGENTS, EVENTS, agent_features, load_episode, prediction_metrics, sha256,
)

ARTIFACTS = ("event_policy.joblib", "next_observation_ridge.joblib", "design.json")
BOUNDARIES = {"formal_score": False, "flight_admitted": False,
              "closed_loop": False, "success": None,
              "world_model_action_conditioned": False,
              "compositional_transfer_established": False}


def complete_test_paths(root, extra=()):
    expected = {condition_key(row) for row in capture_design()["held_out_route_test"]["conditions"]}
    selected = {}
    for path in [*sorted(Path(root).iterdir()), *(Path(p) for p in extra)]:
        # Never resurrect a known failed collection by changing JPEG decoder.
        if (path / "batch_audit_error.txt").exists() or not (path / "inspection.json").is_file():
            continue
        frozen = json.loads((path / "inspection.json").read_text(encoding="utf8"))
        condition = frozen.get("condition")
        if (not frozen.get("passed") or condition is None or not condition.get("reverse")
                or frozen.get("split") != "held_out_route_test"
                or frozen.get("route_id") != "stair_descent_v1"
                or frozen.get("message_delay_s") != .5):
            continue
        target = condition["target"]["color"] + ":" + condition["target"]["shape"]
        key = condition_key({**condition, "target": target})
        if key not in expected or key in selected:
            continue
        fresh = inspect(path)
        if fresh["files_sha256"] != frozen.get("files_sha256"):
            raise ValueError(f"Held-out capture changed: {path}")
        selected[key] = path.resolve()
    if set(selected) != expected:
        raise ValueError(f"Held-out matrix incomplete: {len(selected)}/8 audited conditions; finish collection first")
    return [selected[key] for key in sorted(expected)]


def event_diagnostics(model, episodes):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    expected, predicted = [], []
    for episode in episodes:
        raw = np.asarray(model.predict(agent_features(episode["features"]))).reshape(-1, len(AGENTS))
        if raw.shape[0] != len(episode["events"]) or not np.isin(raw, EVENTS).all():
            raise ValueError("Event predictions must cover every agent decision with known labels")
        for labels, predictions in zip(episode["events"], raw):
            expected.extend(labels[agent] for agent in AGENTS)
            predicted.extend(predictions.tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=list(EVENTS), zero_division=0)
    matrix = confusion_matrix(expected, predicted, labels=list(EVENTS))
    return {"agent_decisions": len(expected),
            "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
            "always_none_accuracy": expected.count("none") / len(expected),
            "macro_f1": float(np.mean(f1)),
            "confusion_matrix_order": list(EVENTS),
            "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "false_event_reports_on_none": int(matrix[0, 1:].sum()),
            "missed_events_predicted_none": int(matrix[1:, 0].sum()),
            "per_event": {name: {"support": int(support[i]),
                "predicted": predicted.count(name), "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i])}
                for i, name in enumerate(EVENTS)}}


class Persistence:
    """Keep the present rendered view and pose as the next prediction."""

    def predict(self, features):
        # public_features starts with 384 image and 14 pose components.
        return np.asarray(features)[:, :398]


def calculate(policy, predictor, episodes):
    per_condition = []
    for episode in episodes:
        condition = episode["report"]["condition"]
        per_condition.append({"episode": str(episode["root"]),
            "condition": {k: condition[k] for k in ("target", "swapped", "teammate", "message_delay_s", "reverse")},
            "event_policy": event_diagnostics(policy, [episode]),
            "next_observation": prediction_metrics(predictor, [episode]),
            "persistence": prediction_metrics(Persistence(), [episode])})
    return {"episodes": len(episodes),
            "event_policy": event_diagnostics(policy, episodes),
            "next_observation": prediction_metrics(predictor, episodes),
            "persistence": prediction_metrics(Persistence(), episodes),
            "per_condition": per_condition, **BOUNDARIES}


def verified_checkpoint(run_dir):
    run_dir = Path(run_dir).resolve()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf8"))
    fingerprints = {name: sha256(run_dir / name) for name in ARTIFACTS}
    if result.get("artifacts_sha256") != fingerprints:
        raise ValueError("Training artifact fingerprint mismatch")
    design = json.loads((run_dir / "design.json").read_text(encoding="utf8"))
    if design.get("event_feature_order") != "shared_self_peer_v1":
        raise ValueError("Frozen matrix evaluation requires shared self/peer features")
    audit = json.loads((run_dir / "independent_audit.json").read_text(encoding="utf8"))
    if audit.get("passed") is not True:
        raise ValueError("Training audit must pass before frozen evaluation")
    # Recheck training provenance against the inventory bound when fitting.
    for path, expected in design["training_sources"].items():
        episode = load_episode(path, "development_training_only")
        if episode["report"]["files_sha256"] != expected:
            raise ValueError("Training source changed after fitting")
    import joblib
    return (run_dir, fingerprints, joblib.load(run_dir / ARTIFACTS[0]),
            joblib.load(run_dir / ARTIFACTS[1]))


def evaluate(run_dir, test_root, extra, output):
    paths = complete_test_paths(test_root, extra)
    run_dir, fingerprints, policy, predictor = verified_checkpoint(run_dir)
    episodes = [load_episode(path, "held_out_route_test") for path in paths]
    report = calculate(policy, predictor, episodes)
    plan = {"schema": "aerowam-subt-frozen-offline-evaluation/1",
            "checkpoint_dir": str(run_dir), "artifacts_sha256": fingerprints,
            "sources": {str(e["root"]): e["report"]["files_sha256"] for e in episodes},
            "test_root": str(Path(test_root).resolve()),
            "extra_test_episodes": [str(Path(p).resolve()) for p in extra],
            "checkpoint_refitted": False, "thresholds_tuned_on_test": False,
            "scope": "eight direction-held-out cells of one scene family; one episode per cell",
            "teacher_forced_recorded_messages": True, **BOUNDARIES}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(json.dumps(plan, indent=2), encoding="utf8")
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def audit(output):
    output = Path(output)
    plan = json.loads((output / "evaluation.json").read_text(encoding="utf8"))
    saved = json.loads((output / "result.json").read_text(encoding="utf8"))
    if any(plan.get(k) != v or saved.get(k) != v for k, v in BOUNDARIES.items()):
        raise ValueError("Offline evaluation boundary changed")
    _, fingerprints, policy, predictor = verified_checkpoint(plan["checkpoint_dir"])
    if fingerprints != plan["artifacts_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    paths = complete_test_paths(plan["test_root"], plan["extra_test_episodes"])
    if set(map(str, paths)) != set(plan["sources"]):
        raise ValueError("Held-out source selection changed")
    episodes = [load_episode(path, "held_out_route_test") for path in paths]
    if any(e["report"]["files_sha256"] != plan["sources"][str(e["root"])] for e in episodes):
        raise ValueError("Held-out source changed after evaluation")
    recalculated = calculate(policy, predictor, episodes)
    if recalculated != saved:
        raise ValueError("Independent frozen evaluation recomputation differs")
    report = {"passed": True, "episodes_recomputed": len(episodes),
              "evaluation_sha256": sha256(output / "evaluation.json"),
              "result_sha256": sha256(output / "result.json"), **BOUNDARIES}
    (output / "independent_audit.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "audit"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--extra-test-episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "evaluate":
        if args.run_dir is None or args.test_root is None:
            parser.error("evaluate needs --run-dir and --test-root")
        report = evaluate(args.run_dir, args.test_root, args.extra_test_episodes, args.output_dir)
    else:
        report = audit(args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
