"""Registry for task-family plugins.

Task implementations register factories here while policy, communication, and
episode contracts remain shared across task families.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .task_api import TaskPlugin


TaskFactory = Callable[[Mapping[str, Any] | None], TaskPlugin]


class TaskRegistry:
    """Explicit registry with deterministic task construction."""

    def __init__(self) -> None:
        self._factories: dict[str, TaskFactory] = {}

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def register(self, task_id: str, factory: TaskFactory) -> None:
        """Register one concrete task ID or family alias."""

        if not task_id or task_id.strip() != task_id:
            raise ValueError("task_id must be a non-empty, trimmed string")
        if not callable(factory):
            raise TypeError("task factory must be callable")
        if task_id in self._factories:
            raise KeyError(f"AeroWAM task already registered: {task_id}")
        self._factories[task_id] = factory

    def create(
        self,
        task_id: str,
        task_spec: Mapping[str, Any] | None = None,
    ) -> TaskPlugin:
        """Create a registered task without importing unrelated families."""

        try:
            factory = self._factories[task_id]
        except KeyError as exc:
            available = ", ".join(self.task_ids) or "<none>"
            raise KeyError(
                f"unknown AeroWAM task: {task_id}; registered: {available}"
            ) from exc
        task = factory(task_spec)
        if not isinstance(task, TaskPlugin):
            raise TypeError(f"factory for {task_id} did not return a TaskPlugin")
        return task


TASK_REGISTRY = TaskRegistry()


def register_task(task_id: str, factory: TaskFactory) -> None:
    """Register a factory in the process-wide task registry."""

    TASK_REGISTRY.register(task_id, factory)


def create_task(
    task_id: str,
    task_spec: Mapping[str, Any] | None = None,
) -> TaskPlugin:
    """Create a task from the process-wide task registry."""

    return TASK_REGISTRY.create(task_id, task_spec)
