"""Task-plugin protocol for adding new dual-UAV task families."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .contract import JointAction


@runtime_checkable
class TaskPlugin(Protocol):
    """Environment boundary shared by mock and ROS/Gazebo task runners.

    A new task family should implement this protocol without changing policy
    backends, communication simulation, or the aggregate evaluator.
    """

    task_id: str
    simulator: str

    def reset(self, seed: int, task_spec: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Reset and return policy-visible observations only."""

    def step(
        self, action: JointAction | Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        """Advance one synchronized control step."""

    def episode_record(self) -> Mapping[str, Any]:
        """Return the evaluator-side, replayable episode record."""


def validate_policy_observation(observation: Mapping[str, Any]) -> None:
    """Reject common truth leakage before a task is handed to a policy."""

    forbidden = {
        "target_pose",
        "target_position",
        "target_identity",
        "teammate_pose",
        "teammate_position",
        "global_map",
        "future_trajectory",
        "truth",
        "hidden_truth",
    }

    def visit(value: Any, path: str = "observation") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in forbidden:
                    raise ValueError(f"policy observation exposes forbidden field: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(observation)

