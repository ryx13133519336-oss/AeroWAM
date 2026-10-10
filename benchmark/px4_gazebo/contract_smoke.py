"""Contract smoke test used before handing the benchmark to a task author."""

from __future__ import annotations

from .contract import Action, HandoffTask, JointAction
from .handoff_env import HandoffEnv
from .policy_api import PolicyContext, PolicyOutput, Prediction
from .task_api import TaskPlugin, validate_policy_observation


def main() -> int:
    task = HandoffTask()
    env = HandoffEnv(task, seed=7)
    observation = env.reset(seed=7)
    assert isinstance(env, TaskPlugin)
    validate_policy_observation(observation)

    context = PolicyContext(
        task=observation["task"],
        agent_id="uav_0",
        observation=observation["observations"]["uav_0"],
        timestamp_ms=observation["timestamp_ms"],
        seed=7,
    )
    output = PolicyOutput(
        action=Action(),
        message={"intent": "observe"},
        prediction=Prediction(horizon_steps=2, target_visible_probability=(0.5, 0.5)),
        confidence=0.5,
    )
    assert context.to_dict()["agent_id"] == "uav_0"
    assert output.to_dict()["action"]["control"] == [0.0, 0.0, 0.0, 0.0]
    next_observation, _, _, info = env.step(JointAction(message=output.message))
    validate_policy_observation(next_observation)
    assert "truth" in info, "truth must stay in evaluator-side info, not observation"
    assert "truth" not in next_observation
    print("policy/task contract smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

