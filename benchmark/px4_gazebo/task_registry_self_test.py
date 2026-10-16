"""ROS-free checks for the task-family registry."""

from __future__ import annotations

from typing import Any, Mapping

from .contract import JointAction
from .task_registry import TaskRegistry


class _DummyTask:
    task_id = "registry_dummy_v1"
    simulator = "mock"

    def __init__(self, task_spec: Mapping[str, Any] | None = None) -> None:
        self.task_spec = dict(task_spec or {})

    def reset(
        self,
        seed: int,
        task_spec: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del seed, task_spec
        return {}

    def step(
        self,
        action: JointAction | Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        del action
        return {}, 0.0, True, {}

    def episode_record(self) -> Mapping[str, Any]:
        return {"task_id": self.task_id}


def main() -> None:
    registry = TaskRegistry()
    registry.register("dummy", lambda spec: _DummyTask(spec))

    task = registry.create("dummy", {"episode_horizon": 12})
    assert task.task_id == "registry_dummy_v1"
    assert task.task_spec == {"episode_horizon": 12}
    assert registry.task_ids == ("dummy",)

    try:
        registry.register("dummy", lambda spec: _DummyTask(spec))
    except KeyError:
        pass
    else:
        raise AssertionError("duplicate task registration was accepted")

    try:
        registry.create("missing")
    except KeyError as exc:
        assert "registered: dummy" in str(exc)
    else:
        raise AssertionError("unknown task creation was accepted")

    print("task registry self-test: PASS")


if __name__ == "__main__":
    main()
