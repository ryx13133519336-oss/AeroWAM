"""Task plugin declarations shared by data, models and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import assert_json_safe, event_name, identifier, unique_strings

TASK_SCHEMA = "aerowam-task/1"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_version: str = "1"
    event_names: tuple[str, ...] = field(default_factory=tuple)
    action_names: tuple[str, ...] = field(default_factory=tuple)
    instruction_fields: tuple[str, ...] = ("instruction",)
    evaluator_id: str = ""

    def __post_init__(self) -> None:
        identifier(self.task_id, "task_id")
        identifier(self.task_version, "task_version")
        if not self.event_names:
            raise ValueError("task must declare at least one event")
        for event in self.event_names:
            event_name(event)
            if not event.startswith(self.task_id + "."):
                raise ValueError("task events must use the task namespace")
        for action in self.action_names:
            identifier(action, "action_name")
        unique_strings(self.instruction_fields, "instruction_fields")
        if self.evaluator_id:
            identifier(self.evaluator_id, "evaluator_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SCHEMA, "task_id": self.task_id, "task_version": self.task_version,
            "event_names": list(self.event_names), "action_names": list(self.action_names),
            "instruction_fields": list(self.instruction_fields), "evaluator_id": self.evaluator_id,
        }


def validate_task(value: Mapping[str, Any]) -> TaskSpec:
    if not isinstance(value, Mapping) or value.get("schema") != TASK_SCHEMA:
        raise ValueError("unsupported task schema")
    expected = {"schema", "task_id", "task_version", "event_names", "action_names", "instruction_fields", "evaluator_id"}
    if set(value) != expected:
        raise ValueError(f"task fields must be exactly {sorted(expected)}")
    assert_json_safe(value)
    return TaskSpec(
        task_id=value["task_id"], task_version=value["task_version"],
        event_names=tuple(value["event_names"]), action_names=tuple(value["action_names"]),
        instruction_fields=tuple(value["instruction_fields"]), evaluator_id=value["evaluator_id"],
    )


HANDOFF_TASK = TaskSpec(
    task_id="handoff", event_names=("handoff.found", "handoff.handoff_request", "handoff.handoff_accept", "handoff.confirm"),
    action_names=("move", "hold", "report"), evaluator_id="handoff_evaluator_v1",
)
SEARCH_TASK = TaskSpec(
    task_id="search", event_names=("search.target_found", "search.target_lost"),
    action_names=("move", "hold", "report"), evaluator_id="search_evaluator_v1",
)
INSPECTION_TASK = TaskSpec(
    task_id="inspection", event_names=("inspection.inspect_started", "inspection.inspect_complete"),
    action_names=("move", "hold", "report"), evaluator_id="inspection_evaluator_v1",
)
