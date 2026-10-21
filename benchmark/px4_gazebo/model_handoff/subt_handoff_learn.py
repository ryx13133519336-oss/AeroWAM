"""Train and independently score exploratory stair event/RGB prediction baselines.

The event baseline is centralized over both agents' public observations. The
world model predicts a low-resolution next rendered observation and proxy pose;
it is not action-conditioned UAV dynamics or PX4 evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .subt_handoff_inspect import inspect, read_rows, safe_image
from .subt_handoff_route import AGENTS

EVENTS = ("none", "found", "handoff_request", "handoff_accept", "confirm")
IMAGE_SIZE = (8, 8)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_features(root, observation):
    values = []
    for agent in AGENTS:
        path = safe_image(root, observation["rgb"][agent])
        with Image.open(path) as frame:
            image = frame.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            values.extend(np.asarray(image, dtype=np.float32).reshape(-1) / 255.)
    return values


def public_features(root, observation):
    image = image_features(root, observation)
    pose = []
    for agent in AGENTS:
        proprio = observation["proprioception"][agent]
        pose.extend(proprio["position_local_m"])
        pose.extend(proprio["quaternion_xyzw"])
    counts = np.zeros((len(AGENTS), len(EVENTS)), dtype=np.float32)
    last = np.zeros_like(counts)
    claims = observation["messages"]
    for index, claim in enumerate(claims):
        counts[AGENTS.index(claim["agent_id"]), EVENTS.index(claim["event"])] += 1.
        last[AGENTS.index(claim["agent_id"]), EVENTS.index(claim["event"])] = index + 1
    recent = last / max(1, len(claims))
    dt = [0. if observation["dt_s"] is None else observation["dt_s"]]
    return np.asarray(image + pose + counts.reshape(-1).tolist() +
                      recent.reshape(-1).tolist() + dt, dtype=np.float32)


def event_labels(supervision):
    labels = []
    for row in supervision:
        per_agent = dict.fromkeys(AGENTS, "none")
        for event in row["events"]:
            per_agent[event["agent_id"]] = event["event"]
        labels.append(per_agent)
    return labels


def agent_features(features):
    """Shared event head sees self/peer ordering, never a fixed vehicle role.

    Layout is two 8x8 RGB views, two seven-component proxy poses, per-agent
    event counts/recency, and dt. Swap every agent-dependent block together.
    """
    features = np.asarray(features)
    image = 8*8*3
    pose_start = 2*image
    count_start = pose_start+14
    recent_start = count_start+2*len(EVENTS)
    if features.ndim != 2 or features.shape[1] != recent_start+2*len(EVENTS)+1:
        raise ValueError("Unexpected public feature layout")
    indices = list(range(features.shape[1]))
    for start, width in ((0,image),(pose_start,7),(count_start,len(EVENTS)),(recent_start,len(EVENTS))):
        indices[start:start+2*width] = list(range(start+width,start+2*width))+list(range(start,start+width))
    return np.stack((features,features[:,indices]),axis=1).reshape(-1,features.shape[1])


def load_episode(root, expected_split):
    root = Path(root).resolve()
    frozen = json.loads((root / "inspection.json").read_text(encoding="utf8"))
    if not frozen.get("passed") or frozen.get("split", "development_training_only") != expected_split:
        raise ValueError(f"Episode is not independently admitted to {expected_split}: {root}")
    fresh = inspect(root)
    if fresh["files_sha256"] != frozen["files_sha256"]:
        raise ValueError(f"Episode changed since frozen independent inspection: {root}")
    observations = read_rows(root / "public/observations.jsonl")
    supervision = read_rows(root / "supervision/labels.jsonl")
    if len(observations) != len(supervision):
        raise ValueError("Public and supervised rows do not align")
    features = np.stack([public_features(root, row) for row in observations])
    images = np.stack([image_features(root, row) for row in observations])
    poses = np.asarray([[value for agent in AGENTS for value in
        (row["proprioception"][agent]["position_local_m"] +
         row["proprioception"][agent]["quaternion_xyzw"])] for row in observations], dtype=np.float32)
    return {"root": root, "report": fresh, "observations": observations,
            "features": features, "images": images, "poses": poses,
            "events": event_labels(supervision)}


def event_metrics(model, episodes, *, self_peer=True):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    expected, predicted = [], []
    for episode in episodes:
        inputs = agent_features(episode["features"]) if self_peer else np.repeat(episode["features"],len(AGENTS),axis=0)
        raw = model.predict(inputs)
        for row, prediction in zip(episode["events"], raw.reshape(-1, len(AGENTS))):
            for agent, value in zip(AGENTS, prediction):
                expected.append(row[agent])
                predicted.append(value)
    labels = list(EVENTS)
    return {"agent_decisions": len(expected), "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
            "per_event": {name: {"support": int(sum(value == name for value in expected)),
                                  "recall": float(recall_score(expected, predicted, labels=[name],
                                                               average="macro", zero_division=0))}
                          for name in labels}}


def prediction_metrics(model, episodes):
    image_errors, pose_errors, pairs = [], [], 0
    for episode in episodes:
        x = episode["features"][:-1]
        actual = np.concatenate((episode["images"][1:], episode["poses"][1:]), axis=1)
        predicted = model.predict(x)
        image_delta = predicted[:, :episode["images"].shape[1]] - actual[:, :episode["images"].shape[1]]
        pose_delta = predicted[:, episode["images"].shape[1]:] - actual[:, episode["images"].shape[1]:]
        image_errors.extend((image_delta ** 2).reshape(-1).tolist())
        pose_errors.extend(np.abs(pose_delta).reshape(-1).tolist())
        pairs += len(x)
    mse = float(np.mean(image_errors))
    return {"next_observation_pairs": pairs, "rgb_8x8_two_view_mse_0_1": mse,
            "rgb_8x8_two_view_rmse_0_1": float(np.sqrt(mse)),
            "proxy_pose_mae_mixed_xyz_quaternion": float(np.mean(pose_errors))}


def complete_development_paths(root, extra=()):
    """Select one freshly verified capture per frozen fixed-communication cell.

    Prefer sorted first successful attempts; never pick a checkpoint using test
    data, substitute a failed condition, or treat repeated frames as repeats.
    """
    from .subt_handoff_design import design
    from .subt_handoff_batch import condition_key
    expected = {condition_key(row) for row in design()["development"]["conditions"]}
    selected = {}
    for path in [*sorted(Path(root).iterdir()),*(Path(p) for p in extra)]:
        if not (path/"inspection.json").is_file():
            continue
        frozen = json.loads((path/"inspection.json").read_text(encoding="utf8"))
        condition = frozen.get("condition")
        if (not frozen.get("passed") or condition is None or condition.get("reverse")
                or frozen.get("split","development_training_only")!="development_training_only"
                or frozen.get("message_delay_s")!=.5):
            continue
        key=condition_key({**condition,"target":condition["target"]["color"]+":"+condition["target"]["shape"]})
        if key not in expected or key in selected:
            continue
        fresh=inspect(path)
        if fresh["files_sha256"]!=frozen["files_sha256"]:
            raise ValueError(f"Development capture changed: {path}")
        selected[key]=path.resolve()
    if set(selected)!=expected:
        raise ValueError(f"Development matrix incomplete: {len(selected)}/8 audited conditions; finish collection first")
    return [selected[key] for key in sorted(expected)]


def train(training_paths, test_path, output):
    if len({Path(path).resolve() for path in training_paths}) != len(training_paths):
        raise ValueError("Repeated training episodes are not independent evidence")
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    training = [load_episode(path, "development_training_only") for path in training_paths]
    test = load_episode(test_path, "held_out_route_test")
    if any(item["report"]["route_id"] != "stair_ascent_v1" for item in training):
        raise ValueError("Training requires development ascent episodes")
    if test["report"]["route_id"] != "stair_descent_v1":
        raise ValueError("Evaluation requires the held-out descent route")
    if any(item["report"].get("condition") is None or item["report"]["message_delay_s"] != .5
           for item in [*training,test]):
        raise ValueError("Current training/evaluation requires modern fixed-0.5s episodes; legacy and 2s data refused")
    x = np.concatenate([item["features"] for item in training])
    y = np.asarray([[item[agent] for agent in AGENTS] for episode in training
                    for item in episode["events"]], dtype=object)
    policy_x = agent_features(x)
    policy_y = y.reshape(-1)
    policy = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=20261021))
    policy.fit(policy_x, policy_y)
    model_x = np.concatenate([item["features"][:-1] for item in training])
    model_y = np.concatenate([np.concatenate((item["images"][1:], item["poses"][1:]), axis=1)
                              for item in training])
    world_model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    world_model.fit(model_x, model_y)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import joblib
    joblib.dump(policy, output / "event_policy.joblib")
    joblib.dump(world_model, output / "next_observation_ridge.joblib")
    sources = {str(item["root"]): item["report"]["files_sha256"] for item in training}
    design = {"schema": "aerowam-subt-stair-learning-baseline/1",
        "training_split": "development_training_only", "evaluation_split": "held_out_route_test",
        "training_route_id": "stair_ascent_v1", "evaluation_route_id": "stair_descent_v1",
        "training_episodes": len(training), "training_samples": int(sum(len(i["features"]) for i in training)),
        "evaluation_episode_count": 1, "evaluation_samples": len(test["features"]),
        "event_policy": "centralized joint-public-observation per-agent multinomial logistic regression",
        "event_feature_order": "shared_self_peer_v1",
        "world_model": "ridge prediction of next 8x8 RGB per view and proxy position/quaternion",
        "features": ["both RGB", "both public proxy poses", "delivered event messages", "dt_s"],
        "privileged_inputs_used": False, "teacher_labels_used_for_supervised_training": True,
        "px4_started": False, "flight_admitted": False, "formal_score": False,
        "evidence_scope": "single held-out episode exploratory diagnostic, same scene family",
        "training_sources": sources,
        "evaluation_source": {str(test["root"]): test["report"]["files_sha256"]}}
    (output / "design.json").write_text(json.dumps(design, indent=2), encoding="utf8")
    result = {"event_policy_train": event_metrics(policy, training),
        "event_policy_held_out_route": event_metrics(policy, [test]),
        "next_observation_train": prediction_metrics(world_model, training),
        "next_observation_held_out_route": prediction_metrics(world_model, [test]),
        "model_score_eligible": False, "formal_score": False, "flight_admitted": False,
        "decentralized_policy_evaluated": False, "world_model_action_conditioned": False,
        "compositional_transfer_established": False,
        "limitations": ["single held-out episode", "same scene family and route geometry",
                         "privileged expert labels", "event policy is centralized",
                         "8x8 rendered RGB proxy is not a learned physical world model"]}
    result["artifacts_sha256"] = {name: sha256(output / name) for name in
        ("event_policy.joblib", "next_observation_ridge.joblib", "design.json")}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def audit(run_dir):
    run_dir = Path(run_dir)
    design = json.loads((run_dir / "design.json").read_text(encoding="utf8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf8"))
    if result.get("artifacts_sha256") != {name: sha256(run_dir / name) for name in
            ("event_policy.joblib", "next_observation_ridge.joblib", "design.json")}:
        raise ValueError("Training artifact fingerprint mismatch")
    training = [load_episode(path, "development_training_only") for path in design["training_sources"]]
    expected_training = {path: report for path, report in design["training_sources"].items()}
    if any(item["report"]["files_sha256"] != expected_training[str(item["root"])]
           for item in training):
        raise ValueError("Development episode changed after model fitting")
    test_path, expected = next(iter(design["evaluation_source"].items()))
    test = load_episode(test_path, "held_out_route_test")
    if test["report"]["files_sha256"] != expected:
        raise ValueError("Held-out evidence changed after training")
    import joblib
    policy = joblib.load(run_dir / "event_policy.joblib")
    world_model = joblib.load(run_dir / "next_observation_ridge.joblib")
    recalculated = {"event_policy_held_out_route": event_metrics(policy, [test],
                        self_peer=design.get("event_feature_order")=="shared_self_peer_v1"),
                    "next_observation_held_out_route": prediction_metrics(world_model, [test])}
    for key, value in recalculated.items():
        if value != result[key]:
            for metric, actual in value.items():
                saved = result[key][metric]
                if isinstance(actual, dict):
                    if actual != saved: raise ValueError(f"Independent recomputation mismatch: {key}.{metric}")
                elif isinstance(actual, float):
                    if abs(actual - saved) > 1e-10: raise ValueError(f"Independent recomputation mismatch: {key}.{metric}")
                elif actual != saved:
                    raise ValueError(f"Independent recomputation mismatch: {key}.{metric}")
    report = {"passed": True, **recalculated, "model_score_eligible": False,
              "formal_score": False, "flight_admitted": False}
    (run_dir / "independent_audit.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--train-episodes", type=Path, nargs="+")
    parser.add_argument("--train-root", type=Path,
                        help="select one audited episode from each of all 8 fixed-communication development conditions")
    parser.add_argument("--extra-train-episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--test-episode", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        if args.train_root is not None:
            if args.train_episodes:parser.error("Use --train-root or --train-episodes, not both")
            args.train_episodes=complete_development_paths(args.train_root,args.extra_train_episodes)
        if not args.train_episodes or args.test_episode is None or args.output_dir is None:
            parser.error("train needs --train-episodes, --test-episode and a new --output-dir")
        answer = train(args.train_episodes, args.test_episode, args.output_dir)
    else:
        if args.run_dir is None: parser.error("audit needs --run-dir")
        answer = audit(args.run_dir)
    print(json.dumps(answer, indent=2), flush=True)


if __name__ == "__main__":
    main()
