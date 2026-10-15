"""Smoke tests for the registered search/handoff task family."""

from __future__ import annotations

from ...contract import Action, JointAction
from ...task_api import validate_policy_observation
from ...task_registry import create_task


def main() -> None:
    task = create_task("search_handoff")
    observation = task.reset(seed=0)
    validate_policy_observation(observation)
    for _ in range(160):
        observation, _, done, _ = task.step(JointAction(uav_0=Action(vx=3), uav_1=Action(vx=3), message={"intent": "found"}))
        validate_policy_observation(observation)
        assert "truth" not in observation
        if done:
            break
    record = task.episode_record()
    assert record["task_id"] == "dual_search_handoff_verify_v1"
    assert record["environment"]["simulator"] == "mock"
    assert record["outcome"]["termination_reason"] in {"timeout", "collision", "success"}
    print("search/handoff task-family self-test: PASS")


if __name__ == "__main__":
    main()
