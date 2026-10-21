"""Self-test for the shared multi-task/multi-scene contracts."""

from __future__ import annotations

import json

from .audit_schema import AuditSummary, validate_audit
from .episode_schema import EpisodeRecord, FutureObservationTarget, MessageRecord, PublicObservation, TaskSupervision, episode_from_dict
from .model_io import ModelInput, ModelOutput, WorldModelOutput
from .scene_schema import SUBT_URBAN_STAIR, validate_scene
from .task_schema import HANDOFF_TASK, INSPECTION_TASK, SEARCH_TASK, validate_task


def run() -> None:
    for task in (HANDOFF_TASK, SEARCH_TASK, INSPECTION_TASK):
        assert validate_task(task.to_dict()).task_id == task.task_id
    assert validate_scene(SUBT_URBAN_STAIR.to_dict()).scene_id == SUBT_URBAN_STAIR.scene_id
    obs = PublicObservation(
        sample_index=0, timestamp_ns=100,
        agents={"uav_0": {"rgb": "images/0_uav_0.jpg", "state": {"position": [0, 0, 0]}},
                 "uav_1": {"rgb": "images/0_uav_1.jpg", "state": {"position": [1, 0, 0]}}},
        task_context={"dt_s": 0.5},
    )
    audit = AuditSummary(True, True, privileged_teacher=True, kinematic_proxy=True, limitations=("diagnostic_only",))
    record = EpisodeRecord(
        "ep_000", "handoff", "1", "subt_urban_stair_v1", "subt_urban_2_story", "development_training_only",
        ("uav_0", "uav_1"), (obs,),
        messages=(MessageRecord(100, "uav_0", "uav_1", {"event": "handoff.found"}),),
        task_supervision=(TaskSupervision(0, "uav_0", "handoff.found"),),
        future_observation_supervision=(FutureObservationTarget(0, 1, 0.5, {"next_rgb": {"uav_0": "images/1_uav_0.jpg"}}),),
        private_audit=audit,
    )
    assert episode_from_dict(record.to_dict()).canonical_bytes() == record.canonical_bytes()
    model_input = ModelInput("ep_000", "handoff", "1", "subt_urban_stair_v1", "subt_urban_2_story", "Track and hand off.", 100, ("uav_0", "uav_1"), obs.agents)
    output = ModelOutput("ep_000", 100, {"uav_0": {"action": "hold"}, "uav_1": {"action": "hold"}})
    assert model_input.to_wire()["schema"].endswith("/1") and output.to_wire()["schema"].endswith("/1")
    assert WorldModelOutput("ep_000", 100, {"rgb": {"uav_0": "predicted.jpg"}}).to_wire()["action_conditioned"] is False
    assert WorldModelOutput("ep_000", 100, {"rgb": {"uav_0": "predicted.jpg"}}, True, {"uav_0": {"action": "hold"}}).to_wire()["action_conditioned"] is True
    assert validate_audit(audit.to_dict()).privileged_teacher
    try:
        PublicObservation(0, 0, {"uav_0": {"target_pose": [0, 0, 0]}, "uav_1": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("truth-bearing public field was accepted")
    print(json.dumps({"passed": True, "tasks": 3, "agents": 2, "score_eligible": False}))


if __name__ == "__main__":
    run()
