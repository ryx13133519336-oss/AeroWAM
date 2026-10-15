"""Registry for task-family plugins.

Adding a task family only requires a package under ``tasks/`` and one registry
entry. Policies, communication, and the episode schema remain shared.
"""

from __future__ import annotations

from typing import Any, Mapping, Callable

from .task_api import TaskPlugin


TaskFactory = Callable[[Mapping[str, Any] | None], TaskPlugin]


def _search_handoff_factory(task_spec: Mapping[str, Any] | None = None) -> TaskPlugin:
    from .tasks.search_handoff.task import SearchHandoffTask

    return SearchHandoffTask(task_spec=task_spec)


TASK_REGISTRY: dict[str, TaskFactory] = {
    "search_handoff": _search_handoff_factory,
}


def create_task(task_id: str, task_spec: Mapping[str, Any] | None = None) -> TaskPlugin:
    """Create a task family by registry key or concrete task ID."""

    factory = TASK_REGISTRY.get(task_id)
    if factory is None:
        for candidate in TASK_REGISTRY.values():
            task = candidate(task_spec)
            if task.task_id == task_id:
                return task
        raise KeyError(f"unknown AeroWAM task: {task_id}")
    return factory(task_spec)
