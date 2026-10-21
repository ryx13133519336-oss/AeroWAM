"""Stable public model and observation-prediction I/O.

No output field assigns a source/receiver role to a vehicle.  A model can
emit per-agent actions and namespaced task events, while the task evaluator
decides their meaning from the observed episode state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import assert_json_safe, assert_public, event_name, finite, identifier, nonnegative_int, unique_strings

MODEL_INPUT_SCHEMA = "aerowam-model-input/1"
MODEL_OUTPUT_SCHEMA = "aerowam-model-output/1"
WORLD_MODEL_INPUT_SCHEMA = "aerowam-world-model-input/1"
WORLD_MODEL_OUTPUT_SCHEMA = "aerowam-world-model-output/1"


@dataclass(frozen=True)
class ModelInput:
    episode_id: str
    task_id: str
    task_version: str
    scene_id: str
    scene_family_id: str
    instruction: str
    timestamp_ns: int
    agent_ids: tuple[str, ...]
    observations: Mapping[str, Mapping[str, Any]]
    messages: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    action_history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    task_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.episode_id, "episode_id"), (self.task_id, "task_id"), (self.task_version, "task_version"), (self.scene_id, "scene_id"), (self.scene_family_id, "scene_family_id")):
            identifier(value, name)
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction is required")
        nonnegative_int(self.timestamp_ns, "timestamp_ns")
        agents = unique_strings(self.agent_ids, "agent_ids", min_items=2)
        if set(self.observations) != set(agents):
            raise ValueError("observations must contain exactly agent_ids")
        for value in self.observations.values():
            if not isinstance(value, Mapping):
                raise ValueError("agent observation must be an object")
        assert_public(self.observations, "model_input.observations")
        assert_public(self.messages, "model_input.messages")
        assert_public(self.action_history, "model_input.action_history")
        assert_public(self.task_context, "model_input.task_context")

    def to_wire(self) -> dict[str, Any]:
        return {"schema": MODEL_INPUT_SCHEMA, "episode_id": self.episode_id, "task_id": self.task_id,
                "task_version": self.task_version, "scene_id": self.scene_id, "scene_family_id": self.scene_family_id,
                "instruction": self.instruction, "timestamp_ns": self.timestamp_ns, "agent_ids": list(self.agent_ids),
                "observations": {key: dict(value) for key, value in self.observations.items()},
                "messages": [dict(item) for item in self.messages], "action_history": [dict(item) for item in self.action_history],
                "task_context": dict(self.task_context)}


@dataclass(frozen=True)
class ModelOutput:
    episode_id: str
    timestamp_ns: int
    actions: Mapping[str, Mapping[str, Any]]
    events: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    predictions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.episode_id, "episode_id")
        nonnegative_int(self.timestamp_ns, "timestamp_ns")
        if not isinstance(self.actions, Mapping) or not self.actions:
            raise ValueError("at least one per-agent action is required")
        for agent, action in self.actions.items():
            identifier(agent, "action.agent_id")
            if not isinstance(action, Mapping):
                raise ValueError("action must be an object")
            assert_json_safe(action, f"action.{agent}")
        for event in self.events:
            if not isinstance(event, Mapping) or "event" not in event:
                raise ValueError("model events require an event field")
            event_name(event["event"], "model event")
            assert_public(event, "model_output.events")
        for name, value in self.predictions.items():
            identifier(name, "prediction")
            if not 0.0 <= finite(value, "prediction") <= 1.0:
                raise ValueError("prediction probabilities must be in [0, 1]")

    def to_wire(self) -> dict[str, Any]:
        return {"schema": MODEL_OUTPUT_SCHEMA, "episode_id": self.episode_id, "timestamp_ns": self.timestamp_ns,
                "actions": {key: dict(value) for key, value in self.actions.items()},
                "events": [dict(item) for item in self.events], "predictions": dict(self.predictions)}


@dataclass(frozen=True)
class WorldModelInput:
    observation: ModelInput
    action: Mapping[str, Mapping[str, Any]] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {"schema": WORLD_MODEL_INPUT_SCHEMA, "observation": self.observation.to_wire(), "action": dict(self.action) if self.action is not None else None}


@dataclass(frozen=True)
class WorldModelOutput:
    episode_id: str
    timestamp_ns: int
    predicted_observation: Mapping[str, Any]
    action_conditioned: bool = False
    action: Mapping[str, Mapping[str, Any]] | None = None
    horizon_s: float = 0.5

    def __post_init__(self) -> None:
        identifier(self.episode_id, "episode_id")
        nonnegative_int(self.timestamp_ns, "timestamp_ns")
        if self.horizon_s <= 0 or self.horizon_s > 60:
            raise ValueError("horizon_s must be in (0, 60]")
        assert_public(self.predicted_observation, "predicted_observation")
        if self.action_conditioned and self.action is None:
            raise ValueError("action-conditioned output requires the action used for prediction")
        if self.action is not None:
            assert_public(self.action, "world_model.action")

    def to_wire(self) -> dict[str, Any]:
        return {"schema": WORLD_MODEL_OUTPUT_SCHEMA, "episode_id": self.episode_id,
                "timestamp_ns": self.timestamp_ns, "predicted_observation": dict(self.predicted_observation),
                "action_conditioned": self.action_conditioned,
                "action": dict(self.action) if self.action is not None else None,
                "horizon_s": self.horizon_s}
