"""First useful warehouse task family for cooperation and prediction transfer.

This module is an offline, deterministic task adapter.  It generates public
two-agent observations, private future supervision and a separate truth
branch without changing the shared core schema.  Gazebo/PX4 adapters can use
the same episode config and truth contract later; this module never treats
synthetic kinematics as flight evidence or a model score.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core.audit_schema import AuditSummary, Provenance
from .core.episode_schema import (
    EpisodeRecord, FutureObservationTarget, MessageRecord,
    PublicObservation, TaskSupervision,
)
from .core.model_io import ModelInput
from .core.scene_schema import SceneSpec, validate_scene
from .core.task_schema import TaskSpec, validate_task


TASK_ID = "warehouse_coop_prediction"
TASK_VERSION = "1"
TASK = TaskSpec(
    task_id=TASK_ID,
    task_version=TASK_VERSION,
    event_names=tuple(f"{TASK_ID}.{name}" for name in (
        "observe", "predict", "request", "wait", "confirm", "reject", "return")),
    action_names=("predict", "request", "wait", "confirm", "reject", "hold"),
    instruction_fields=("instruction", "prediction_horizons_s", "message_budget"),
    evaluator_id="warehouse_coop_prediction_evaluator_v1",
)
SCENE = SceneSpec(
    scene_id="aws_warehouse_handoff_v2",
    scene_family_id="aws_warehouse_v2",
    scene_version="2",
    adapter_id="warehouse_v2_scene_task_adapter_v1",
    public_channels=("dual_rgb", "local_state", "visual_track", "messages", "prediction_history"),
    safety_contract="warehouse_spawn_bound_regions_v2",
    supports_tasks=(TASK_ID,),
)

AGENTS = ("uav_0", "uav_1")
HORIZONS_S = (0.5, 1.0, 2.0)
DT_S = 0.5
SAMPLES = 41
MOTION_FAMILIES = ("constant_velocity", "accelerate_brake", "pause_reverse", "branch", "distractor")
LAYOUT_FAMILIES = ("aisle_c", "reverse_c", "l_turn", "staggered", "open_left", "open_right")
TEAMMATE_FAMILIES = ("reliable_relay", "delayed_relay", "conservative_relay", "noisy_relay", "missing_relay")

SPLIT_MATRIX = {
    "seen": ("aisle_c", "constant_velocity", "reliable_relay"),
    "new_layout": ("reverse_c", "constant_velocity", "reliable_relay"),
    "new_motion": ("aisle_c", "pause_reverse", "reliable_relay"),
    "new_teammate": ("aisle_c", "constant_velocity", "delayed_relay"),
    "composed": ("l_turn", "branch", "noisy_relay"),
}

LAYOUTS = {
    "aisle_c": {"blind_low": -0.65, "blind_high": 0.65, "visible_margin": 0.12},
    "reverse_c": {"blind_low": -0.50, "blind_high": 0.72, "visible_margin": 0.10},
    "l_turn": {"blind_low": -0.78, "blind_high": 0.50, "visible_margin": 0.14},
    "staggered": {"blind_low": -0.58, "blind_high": 0.58, "visible_margin": 0.08},
    "open_left": {"blind_low": -0.42, "blind_high": 0.78, "visible_margin": 0.10},
    "open_right": {"blind_low": -0.78, "blind_high": 0.42, "visible_margin": 0.10},
}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _motion(t: float, family: str, branch: int) -> tuple[float, float]:
    """Return hidden target y and velocity, in the warehouse ENU frame."""
    if family == "constant_velocity":
        y = _clip(-1.75 + 0.19 * t, -1.75, 1.67)
        return y, 0.19 if t < 18.0 else 0.0
    if family == "accelerate_brake":
        if t < 6.0:
            return -1.70 + 0.08 * t * t, 0.16 * t
        if t < 12.0:
            u = t - 6.0
            return -1.70 + 2.88 + 0.16 * u - 0.01 * u * u, 0.16 - 0.02 * u
        y = _clip(0.0 + 0.11 * (t - 12.0), -1.70, 1.32)
        return y, 0.11 if t < 24.0 else 0.0
    if family == "pause_reverse":
        if t < 7.0:
            return -1.70 + 0.18 * t, 0.18
        if t < 9.0:
            return -0.44, 0.0
        if t < 12.0:
            u = t - 9.0
            return -0.44 - 0.12 * u, -0.12
        u = t - 12.0
        return _clip(-0.80 + 0.20 * u, -0.80, 1.60), 0.20 if u < 12.0 else 0.0
    if family == "branch":
        if t < 7.0:
            return -1.65 + 0.20 * t, 0.20
        if t < 10.0:
            return -0.25 + 0.04 * (t - 7.0), 0.04
        # The branch is chosen while the target is inside the blind region.
        speed = 0.19 if branch > 0 else 0.15
        return _clip(-0.13 + branch * speed * (t - 10.0), -0.20, 1.65), branch * speed
    if family == "distractor":
        # The target follows a low-speed lane while a separate distractor
        # moves through the other lane. Distractor truth stays private.
        return _clip(-1.15 + 0.07 * t, -1.15, 0.95), 0.07 if t < 30.0 else 0.0
    raise ValueError(f"unknown motion family: {family}")


def _distractor_motion(t: float, branch: int) -> tuple[float, float]:
    phase = 0.11 if branch > 0 else -0.09
    y = _clip(1.35 - phase * t, -0.95, 1.35)
    return y, -phase if -0.95 < y < 1.35 else 0.0


def _visibility(y: float, layout: str) -> tuple[bool, bool]:
    bounds = LAYOUTS[layout]
    return y <= bounds["blind_low"], y >= bounds["blind_high"]


def _instruction() -> str:
    return (
        "Track the moving parcel from partial observations. Predict its future "
        "position and visibility at 0.5s, 1.0s and 2.0s. Request a relay only "
        "when the teammate can add grounded evidence; wait or reject when evidence is insufficient."
    )


def _teammate_messages(profile: str, index: int, timestamp_ns: int,
                       visible: tuple[bool, bool], y: float, rng: random.Random) -> list[MessageRecord]:
    """Generate public teammate behavior; latent profile remains private."""
    receiver_visible = visible[1]
    if not receiver_visible:
        return []
    if profile == "delayed_relay" and index < 2:
        return []
    if profile == "missing_relay" and index % 9 in (0, 1, 2):
        return []
    if profile == "conservative_relay" and rng.random() < 0.25:
        return []
    if profile == "noisy_relay" and index % 11 == 0:
        payload = {"event": "track", "confidence": 0.42, "bearing_y": round(y + 0.55, 3)}
    else:
        payload = {"event": "track", "confidence": 0.91, "bearing_y": round(y + rng.uniform(-0.04, 0.04), 3)}
    return [MessageRecord(timestamp_ns, "uav_1", "uav_0", payload)]


def _truth_target(index: int, family: str, layout: str, branch: int) -> dict[str, Any]:
    t = index * DT_S
    y, velocity = _motion(t, family, branch)
    visible = _visibility(y, layout)
    horizons = {}
    for horizon in HORIZONS_S:
        future_y, future_v = _motion(t + horizon, family, branch)
        future_visible = _visibility(future_y, layout)
        horizons[str(horizon)] = {
            "position_enu": [2.4, future_y, 1.4],
            "velocity_enu": [0.0, future_v, 0.0],
            "visible_by_agent": {AGENTS[0]: future_visible[0], AGENTS[1]: future_visible[1]},
        }
    result = {
        "position_enu": [2.4, y, 1.4], "velocity_enu": [0.0, velocity, 0.0],
        "visible_by_agent": {AGENTS[0]: visible[0], AGENTS[1]: visible[1]},
        "occluded": not any(visible), "horizons": horizons,
    }
    if family == "distractor":
        distractor_y, distractor_v = _distractor_motion(t, branch)
        result["private_distractor"] = {
            "position_enu": [2.4, distractor_y, 1.4],
            "velocity_enu": [0.0, distractor_v, 0.0],
        }
    return result


def generate_episode(seed: int, split: str = "seen", *, with_supervision: bool = True) -> tuple[EpisodeRecord, dict[str, Any]]:
    """Create one public core episode and its private evaluator truth."""
    if split not in SPLIT_MATRIX:
        raise ValueError(f"unknown split: {split}")
    layout, motion, teammate = SPLIT_MATRIX[split]
    rng = random.Random(seed)
    branch = 1 if rng.random() >= 0.5 else -1
    episode_id = f"warehouse_v2_{split}_{seed:08d}"
    observations: list[PublicObservation] = []
    messages: list[MessageRecord] = []
    supervision: list[TaskSupervision] = []
    futures: list[FutureObservationTarget] = []
    truth_rows: list[dict[str, Any]] = []
    seen_messages: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENTS}
    for index in range(SAMPLES):
        stamp = round(index * DT_S * 1e9)
        truth = _truth_target(index, motion, layout, branch)
        y = truth["position_enu"][1]
        visible = tuple(truth["visible_by_agent"][agent] for agent in AGENTS)
        current_messages = _teammate_messages(teammate, index, stamp, visible, y, rng)
        messages.extend(current_messages)
        for message in current_messages:
            seen_messages[message.receiver].append(message.to_dict())
        agents = {}
        for agent_index, agent in enumerate(AGENTS):
            own_visible = visible[agent_index]
            noise = rng.uniform(-0.04, 0.04) if own_visible else None
            agents[agent] = {
                "rgb": f"warehouse_v2/{episode_id}/{index:04d}_{agent}.jpg",
                "local_state": {"self_pose": [0.25, -2.1 if agent_index == 0 else 2.1, 1.75],
                                "health": "nominal"},
                "visual_track": {"visible": own_visible,
                                  "bearing_y": None if noise is None else round(y + noise, 3),
                                  "confidence": 0.88 if own_visible else 0.0},
                "peer_messages": list(seen_messages[agent]),
            }
        context = {
            "role_mode": "unassigned", "prediction_horizons_s": list(HORIZONS_S),
            "message_budget": 8, "observation_semantics": "public_rgb_and_grounded_local_track",
            "teammate_messages_are_claims": True,
        }
        observations.append(PublicObservation(index, stamp, agents, context))
        if with_supervision:
            supervision.append(TaskSupervision(index, "uav_0", f"{TASK_ID}.predict", source="expert"))
            supervision.append(TaskSupervision(index, "uav_1", f"{TASK_ID}.predict", source="expert"))
            if visible[0] and not visible[1]:
                supervision.append(TaskSupervision(index, "uav_0", f"{TASK_ID}.request", source="expert"))
            elif visible[1] and not visible[0]:
                supervision.append(TaskSupervision(index, "uav_1", f"{TASK_ID}.confirm", source="expert"))
            elif not any(visible):
                supervision.append(TaskSupervision(index, "uav_0", f"{TASK_ID}.wait", source="expert"))
        if index < SAMPLES - 1:
            next_truth = _truth_target(index + 1, motion, layout, branch)
            futures.append(FutureObservationTarget(
                index, index + 1, DT_S,
                {"horizons": truth["horizons"], "next_visibility": next_truth["visible_by_agent"]},
                scope="warehouse_v2_private_future_state_and_visibility",
            ))
        truth_rows.append({"sample_index": index, "timestamp_ns": stamp, **truth})
    audit = AuditSummary(
        passed=True, independent_audit=False, model_score_eligible=False, flight_admitted=False,
        privileged_teacher=with_supervision, kinematic_proxy=True, action_conditioned_world_model=False,
        checks={"public_private_split_validated": True, "episode_seed_recorded": True,
                "not_flight_evidence": True},
        limitations=("synthetic_kinematics_not_flight_evidence", "teacher_supervision_for_training_only",
                     "model_score_requires_independent_replay"),
        audit_id="warehouse_v2_synthetic_audit_v1",
    )
    record = EpisodeRecord(
        episode_id=episode_id, task_id=TASK_ID, task_version=TASK_VERSION,
        scene_id=SCENE.scene_id, scene_family_id=SCENE.scene_family_id,
        split=split, agent_ids=AGENTS, observations=tuple(observations),
        messages=tuple(messages), task_supervision=tuple(supervision),
        future_observation_supervision=tuple(futures), private_audit=audit,
        provenance=(Provenance(
            source_id=episode_id, source_kind="warehouse_v2_synthetic_episode",
            capture_tool="warehouse_v2_task_generator",
            notes=("public_excludes_latent_families", "truth_is_private", "no_flight_score"),
        ),),
    )
    truth = {"episode_id": episode_id, "seed": seed, "split": split,
             "layout_family": layout, "motion_family": motion,
             "teammate_family": teammate, "branch": branch, "rows": truth_rows}
    return record, truth


def model_inputs(record: EpisodeRecord) -> list[ModelInput]:
    """Map an episode to public-only model inputs."""
    validate_task(TASK.to_dict())
    validate_scene(SCENE.to_dict())
    messages_by_stamp: dict[int, list[dict[str, Any]]] = {}
    for message in record.messages:
        messages_by_stamp.setdefault(message.timestamp_ns, []).append(message.to_dict())
    result = []
    for observation in record.observations:
        prior = [message.to_dict() for message in record.messages if message.timestamp_ns < observation.timestamp_ns]
        result.append(ModelInput(
            record.episode_id, record.task_id, record.task_version, record.scene_id,
            record.scene_family_id, _instruction(), observation.timestamp_ns, record.agent_ids,
            observation.agents, tuple(prior), (), observation.task_context,
        ))
    return result


def _gaussian_nll(value: float, mean: float, std: float) -> float:
    std = _clip(abs(std), 0.03, 3.0)
    return 0.5 * math.log(2.0 * math.pi * std * std) + ((value - mean) ** 2) / (2.0 * std * std)


def _ece(confidences: list[float], labels: list[bool], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for bucket in range(bins):
        low, high = bucket / bins, (bucket + 1) / bins
        selected = [i for i, value in enumerate(confidences)
                    if low <= value < high or (bucket == bins - 1 and value == high)]
        if selected:
            accuracy = sum(labels[i] for i in selected) / len(selected)
            confidence = sum(confidences[i] for i in selected) / len(selected)
            error += len(selected) / total * abs(accuracy - confidence)
    return error


def score_predictions(record: EpisodeRecord, truth: Mapping[str, Any], predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score future distributions and grounded collaboration actions.

    Predictions are public JSON rows.  Truth is supplied separately by the
    trusted evaluator and must never be passed to a model adapter.
    """
    rows = {int(row["sample_index"]): row for row in predictions}
    truth_rows = {int(row["sample_index"]): row for row in truth["rows"]}
    nll, brier, confidences, labels = [], [], [], []
    unsafe, requests, confirms, successful = 0, 0, 0, False
    for index, target in enumerate(record.future_observation_supervision):
        row = rows.get(target.sample_index)
        actual = truth_rows.get(target.sample_index)
        if row is None or actual is None:
            continue
        horizons = row.get("horizons", {})
        for horizon in HORIZONS_S:
            key = str(horizon)
            expected = actual["horizons"][key]
            predicted = horizons.get(key, {})
            nll.append(_gaussian_nll(expected["position_enu"][1], predicted.get("mean_y", 0.0), predicted.get("std_y", 1.0)))
            probs = predicted.get("visible_probability", {})
            for agent in AGENTS:
                label = bool(expected["visible_by_agent"][agent])
                probability = _clip(float(probs.get(agent, 0.5)), 0.0, 1.0)
                brier.append((probability - float(label)) ** 2)
                confidences.append(probability if label else 1.0 - probability)
                labels.append(label == (probability >= 0.5))
        action = row.get("action", {})
        event = action.get("event")
        if event == "request":
            requests += 1
            # A request is grounded only when the source is occluded and the
            # teammate has independent visibility to add evidence.
            if actual["visible_by_agent"][AGENTS[0]] or not actual["visible_by_agent"][AGENTS[1]]:
                unsafe += 1
        elif event == "confirm":
            confirms += 1
            if actual["visible_by_agent"][AGENTS[1]] and requests:
                successful = True
    return {
        "schema": "aerowam-warehouse-v2-score/1", "episode_id": record.episode_id,
        "prediction_samples": len(nll) // len(HORIZONS_S) if nll else 0,
        "prediction_nll": sum(nll) / len(nll) if nll else None,
        "visibility_brier": sum(brier) / len(brier) if brier else None,
        "calibration_error": _ece(confidences, labels),
        "unsafe_handoff_rate": unsafe / requests if requests else 0.0,
        "handoff_success": bool(successful and unsafe == 0),
        "communication_events": requests + confirms,
        "model_score_eligible": False,
        "private_truth_used_by_evaluator": True,
        "flight_verified": False,
    }


def export_dataset(output: Path, *, episodes_per_split: int = 2, seed: int = 20261021) -> dict[str, Any]:
    if episodes_per_split < 1:
        raise ValueError("episodes_per_split must be positive")
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    private = output / "private_truth"
    private.mkdir()
    public_records, inputs = [], []
    for split_index, split in enumerate(SPLIT_MATRIX):
        for offset in range(episodes_per_split):
            record, truth = generate_episode(seed + split_index * 1000 + offset, split)
            # Keep the downloadable episode index public.  The full core
            # record contains private future supervision and must not be
            # shipped beside model inputs.
            public_records.append({
                "schema": "aerowam-warehouse-v2-public-episode-index/1",
                "episode_id": record.episode_id,
                "task_id": record.task_id,
                "task_version": record.task_version,
                "scene_id": record.scene_id,
                "scene_family_id": record.scene_family_id,
                "split": record.split,
                "agent_ids": list(record.agent_ids),
                "observation_count": len(record.observations),
                "model_input_file": "model_inputs.jsonl",
                "model_score_eligible": False,
            })
            inputs.extend(item.to_wire() for item in model_inputs(record))
            (private / f"{record.episode_id}.json").write_text(json.dumps(truth, indent=2), encoding="utf8")
    (output / "episodes.jsonl").write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in public_records), encoding="utf8")
    (output / "model_inputs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in inputs), encoding="utf8")
    manifest = {
        "schema": "aerowam-warehouse-v2-dataset/2", "public_episode_index_schema": "aerowam-warehouse-v2-public-episode-index/1",
        "task": TASK.to_dict(), "scene": SCENE.to_dict(),
        "splits": {split: episodes_per_split for split in SPLIT_MATRIX},
        "episodes": len(public_records), "model_inputs": len(inputs),
        "private_truth_dir": "private_truth", "model_score_eligible": False,
        "flight_admitted": False, "core_schema_modified": False,
        "public_episode_index_has_private_truth": False,
        "limitations": ["synthetic_kinematics", "no_px4_flight", "truth_separate_from_public_inputs"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    print(json.dumps(export_dataset(args.output, episodes_per_split=args.episodes_per_split, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
