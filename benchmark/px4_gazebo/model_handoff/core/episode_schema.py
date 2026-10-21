"""Task- and scene-neutral episode container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import assert_json_safe, assert_public, event_name, identifier, nonnegative_int, unique_strings
from .audit_schema import AuditSummary, Provenance, validate_audit

EPISODE_SCHEMA = "aerowam-episode/1"


@dataclass(frozen=True)
class PublicObservation:
    sample_index: int
    timestamp_ns: int
    agents: Mapping[str, Mapping[str, Any]]
    task_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonnegative_int(self.sample_index, "sample_index")
        nonnegative_int(self.timestamp_ns, "timestamp_ns")
        if not isinstance(self.agents, Mapping) or len(self.agents) < 2:
            raise ValueError("at least two agent observations are required")
        for agent, value in self.agents.items():
            identifier(agent, "agent_id")
            if not isinstance(value, Mapping):
                raise ValueError("agent observation must be an object")
        assert_public(self.agents, "observation.agents")
        assert_public(self.task_context, "observation.task_context")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_index": self.sample_index, "timestamp_ns": self.timestamp_ns,
                "agents": {key: dict(value) for key, value in self.agents.items()},
                "task_context": dict(self.task_context)}


@dataclass(frozen=True)
class MessageRecord:
    timestamp_ns: int
    sender: str
    receiver: str
    payload: Mapping[str, Any]
    delivered: bool = True

    def __post_init__(self) -> None:
        nonnegative_int(self.timestamp_ns, "message.timestamp_ns")
        identifier(self.sender, "message.sender")
        identifier(self.receiver, "message.receiver")
        if self.sender == self.receiver:
            raise ValueError("message sender and receiver must differ")
        assert_public(self.payload, "message.payload")

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp_ns": self.timestamp_ns, "sender": self.sender, "receiver": self.receiver,
                "payload": dict(self.payload), "delivered": self.delivered}


@dataclass(frozen=True)
class TaskSupervision:
    sample_index: int
    agent_id: str
    event: str
    source: str = "expert"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonnegative_int(self.sample_index, "supervision.sample_index")
        identifier(self.agent_id, "supervision.agent_id")
        event_name(self.event, "supervision.event")
        identifier(self.source, "supervision.source")
        assert_json_safe(self.metadata, "supervision.metadata")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_index": self.sample_index, "agent_id": self.agent_id, "event": self.event,
                "source": self.source, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class FutureObservationTarget:
    sample_index: int
    next_sample_index: int
    delta_s: float
    target: Mapping[str, Any]
    scope: str = "observation_prediction_only"

    def __post_init__(self) -> None:
        nonnegative_int(self.sample_index, "future.sample_index")
        nonnegative_int(self.next_sample_index, "future.next_sample_index")
        if self.next_sample_index <= self.sample_index:
            raise ValueError("future target must point forward")
        if not 0.0 < float(self.delta_s) <= 60.0:
            raise ValueError("future.delta_s must be in (0, 60]")
        identifier(self.scope, "future.scope")
        assert_json_safe(self.target, "future.target")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_index": self.sample_index, "next_sample_index": self.next_sample_index,
                "delta_s": float(self.delta_s), "target": dict(self.target), "scope": self.scope}


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    task_id: str
    task_version: str
    scene_id: str
    scene_family_id: str
    split: str
    agent_ids: tuple[str, ...]
    observations: tuple[PublicObservation, ...]
    messages: tuple[MessageRecord, ...] = field(default_factory=tuple)
    task_supervision: tuple[TaskSupervision, ...] = field(default_factory=tuple)
    future_observation_supervision: tuple[FutureObservationTarget, ...] = field(default_factory=tuple)
    role_assignment: None = None
    private_audit: AuditSummary | None = None
    provenance: tuple[Provenance, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value, name in ((self.episode_id, "episode_id"), (self.task_id, "task_id"), (self.task_version, "task_version"), (self.scene_id, "scene_id"), (self.scene_family_id, "scene_family_id"), (self.split, "split")):
            identifier(value, name)
        agents = unique_strings(self.agent_ids, "agent_ids", min_items=2)
        if self.role_assignment is not None:
            raise ValueError("role_assignment must remain null; roles are formed from observations")
        if not self.observations:
            raise ValueError("episode requires observations")
        indices = [row.sample_index for row in self.observations]
        if indices != list(range(len(indices))):
            raise ValueError("observations require contiguous sample indices")
        stamps = [row.timestamp_ns for row in self.observations]
        if any(right <= left for left, right in zip(stamps, stamps[1:])):
            raise ValueError("observation timestamps must increase")
        if any(set(row.agents) != set(agents) for row in self.observations):
            raise ValueError("each observation must contain exactly agent_ids")
        if any(self.task_id + "." not in row.event for row in self.task_supervision):
            raise ValueError("task supervision event namespace does not match task_id")
        if any(message.sender not in agents or message.receiver not in agents for message in self.messages):
            raise ValueError("message endpoint is not in agent_ids")
        if any(left.timestamp_ns > right.timestamp_ns for left, right in zip(self.messages, self.messages[1:])):
            raise ValueError("messages must be ordered by timestamp")
        if self.private_audit is not None and not isinstance(self.private_audit, AuditSummary):
            raise ValueError("private_audit must be AuditSummary")
        if any(not isinstance(item, Provenance) for item in self.provenance):
            raise ValueError("provenance entries must be Provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EPISODE_SCHEMA, "episode_id": self.episode_id, "task_id": self.task_id,
            "task_version": self.task_version, "scene_id": self.scene_id,
            "scene_family_id": self.scene_family_id, "split": self.split,
            "agent_ids": list(self.agent_ids), "role_assignment": None,
            "observations": [row.to_dict() for row in self.observations],
            "messages": [row.to_dict() for row in self.messages],
            "task_supervision": [row.to_dict() for row in self.task_supervision],
            "future_observation_supervision": [row.to_dict() for row in self.future_observation_supervision],
            "private_audit": self.private_audit.to_dict() if self.private_audit else None,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    def canonical_bytes(self) -> bytes:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def episode_from_dict(value: Mapping[str, Any]) -> EpisodeRecord:
    expected = {"schema", "episode_id", "task_id", "task_version", "scene_id", "scene_family_id", "split", "agent_ids", "role_assignment", "observations", "messages", "task_supervision", "future_observation_supervision", "private_audit", "provenance"}
    if not isinstance(value, Mapping) or value.get("schema") != EPISODE_SCHEMA or set(value) != expected:
        raise ValueError("invalid episode schema or fields")
    if value["role_assignment"] is not None:
        raise ValueError("role_assignment must be null")
    observations = tuple(PublicObservation(row["sample_index"], row["timestamp_ns"], row["agents"], row.get("task_context", {})) for row in value["observations"])
    messages = tuple(MessageRecord(row["timestamp_ns"], row["sender"], row["receiver"], row["payload"], row.get("delivered", True)) for row in value["messages"])
    supervision = tuple(TaskSupervision(row["sample_index"], row["agent_id"], row["event"], row.get("source", "expert"), row.get("metadata", {})) for row in value["task_supervision"])
    futures = tuple(FutureObservationTarget(row["sample_index"], row["next_sample_index"], row["delta_s"], row["target"], row.get("scope", "observation_prediction_only")) for row in value["future_observation_supervision"])
    audit = validate_audit(value["private_audit"]) if value["private_audit"] is not None else None
    provenance = tuple(Provenance(item["source_id"], item["source_kind"], item.get("files_sha256", {}), item.get("parent_episode_id", ""), item.get("capture_tool", ""), tuple(item.get("notes", []))) for item in value["provenance"])
    return EpisodeRecord(value["episode_id"], value["task_id"], value["task_version"], value["scene_id"], value["scene_family_id"], value["split"], tuple(value["agent_ids"]), observations, messages, supervision, futures, None, audit, provenance)
