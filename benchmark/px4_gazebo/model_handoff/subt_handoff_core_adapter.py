"""Adapt an independently audited SubT stair export to the shared core schema.

This is an offline format adapter.  It does not run Gazebo, PX4 or a policy,
and it intentionally retains teacher/proxy limitations in the private audit
branch so the resulting file cannot be mistaken for flight evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .core.audit_schema import AuditSummary, Provenance
from .core.episode_schema import EpisodeRecord, FutureObservationTarget, PublicObservation, TaskSupervision


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _event_name(value: str) -> str | None:
    if value in {"", "none", "null"}:
        return None
    return value if value.startswith("handoff.") else f"handoff.{value}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt(export_dir: Path, output: Path) -> dict[str, Any]:
    """Convert one ``subt_handoff_data.export`` directory to core JSON."""
    export_dir = Path(export_dir).resolve()
    design = json.loads((export_dir / "design.json").read_text(encoding="utf-8"))
    if design.get("schema") != "aerowam-subt-stair-training-export/1":
        raise ValueError("unsupported SubT export schema")
    if design.get("role_assignment") is not None:
        raise ValueError("SubT export must be role-unassigned")
    inputs = _read_jsonl(export_dir / "inputs.jsonl")
    labels = {row["key"]: row for row in _read_jsonl(export_dir / "event_supervision.jsonl")}
    futures = {row["key"]: row for row in _read_jsonl(export_dir / "future_observation_supervision.jsonl")}
    groups = _read_jsonl(export_dir / "groups.jsonl")
    if not inputs or len(inputs) != len(groups):
        raise ValueError("input/group rows must be non-empty and aligned")

    by_episode: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, group in zip(inputs, groups):
        key = row["key"]
        if labels.get(key) is None or group.get("key") != key:
            raise ValueError("missing or misaligned event supervision")
        by_episode.setdefault(group["parent_episode_id"], []).append((row, group))

    records: list[EpisodeRecord] = []
    for episode_id, rows in sorted(by_episode.items()):
        rows.sort(key=lambda item: item[0]["key"])
        observations: list[PublicObservation] = []
        supervision: list[TaskSupervision] = []
        future_targets: list[FutureObservationTarget] = []
        elapsed_s = 0.0
        for index, (row, group) in enumerate(rows):
            observation = row["observation"]
            if index > 0 and observation["dt_s"] is not None:
                elapsed_s += float(observation["dt_s"])
            rgb = observation["rgb"]
            proprioception = observation["proprioception"]
            messages = observation["messages"]
            agents = {
                agent: {"rgb": rgb[agent], "proprioception": proprioception[agent], "messages": messages}
                for agent in ("uav_0", "uav_1")
            }
            # The export has no absolute ROS stamps.  Reconstruct a relative
            # timeline from inter-frame intervals; never invent a sensor stamp.
            observations.append(PublicObservation(index, round(elapsed_s * 1e9), agents, {"dt_s": observation["dt_s"]}))
            for agent, event in labels[row["key"]]["joint_event"].items():
                name = _event_name(event)
                if name is not None:
                    supervision.append(TaskSupervision(index, agent, name, source="expert"))
            target = futures.get(row["key"])
            if target is not None:
                future_targets.append(FutureObservationTarget(
                    index, index + 1, target["dt_s"],
                    {"next_rgb": target["next_rgb"]},
                    scope="observation_prediction_only_no_causal_action_conditioning",
                ))

        audit = AuditSummary(
            passed=True, independent_audit=True, model_score_eligible=False, flight_admitted=False,
            privileged_teacher=True, kinematic_proxy=True, action_conditioned_world_model=False,
            checks={"source_export_declared": True, "role_unassigned": True, "scene_independent_audit": True},
            limitations=(
                "expert_event_labels_are_teacher_supervision",
                "camera_and_pose_sequence_uses_kinematic_proxy",
                "future_observation_prediction_is_not_action_conditioned",
                "same_scene_family_development_export",
            ), audit_id="subt_handoff_core_adapter_v1",
        )
        source_files = {name: _sha256(export_dir / name) for name in (
            "design.json", "inputs.jsonl", "event_supervision.jsonl", "future_observation_supervision.jsonl", "groups.jsonl"
        )}
        record = EpisodeRecord(
            episode_id=episode_id, task_id="handoff", task_version="1",
            scene_id="subt_urban_stair_v1", scene_family_id="subt_urban_2_story",
            split=rows[0][1].get("split", "development_training_only"),
            agent_ids=("uav_0", "uav_1"), observations=tuple(observations),
            task_supervision=tuple(supervision), future_observation_supervision=tuple(future_targets),
            private_audit=audit,
            provenance=(Provenance(
                source_id=episode_id, source_kind="subt_stair_training_export",
                files_sha256=source_files, capture_tool="subt_handoff_data.export",
                notes=("public_inputs_exclude_target_truth", "no_px4_flight", "timestamp_reconstructed_from_dt"),
            ),),
        )
        records.append(record)

    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=True, allow_nan=False) + "\n")
    manifest = {
        "schema": "aerowam-core-adapter-export/1", "source_schema": design["schema"],
        "task_id": "handoff", "scene_id": "subt_urban_stair_v1", "scene_family_id": "subt_urban_2_story",
        "episodes": len(records), "samples": sum(len(item.observations) for item in records),
        "model_score_eligible": False, "flight_admitted": False,
        "action_conditioned_world_model": False,
        "limitations": ["teacher/proxy supervision retained only for diagnosis", "no independent scene in this export"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(adapt(args.export, args.output), ensure_ascii=True))


if __name__ == "__main__":
    main()
