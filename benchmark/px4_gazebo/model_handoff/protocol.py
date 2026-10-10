"""Truth-free policy protocol for learned dual-UAV Handoff policies."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence

from ..contract import AGENT_IDS, Action, JointAction


MODEL_PROTOCOL_VERSION = "aerowam-model-handoff/1"
ACTION_LIMITS = {"dx": 1.0, "dy": 1.0, "dz": 0.5, "dyaw": math.radians(15.0)}
INTENTS = ("none", "handoff", "confirm")
FORBIDDEN_POLICY_KEYS = frozenset({
    "truth", "target_pose", "target_position", "target_id", "target_identity",
    "occluder", "occluder_center", "occluder_half_extents", "global_map",
    "map", "future", "future_trajectory", "evaluator", "fixture_truth",
})


def _finite_tuple(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must have {size} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def assert_truth_free(value: Any, path: str = "policy_input") -> None:
    """Reject evaluator-only fields at any depth before policy serialization."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_POLICY_KEYS:
                raise ValueError(f"forbidden policy field at {path}.{key}")
            assert_truth_free(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_truth_free(item, f"{path}[{index}]")


@dataclass(frozen=True)
class VehicleState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    health: str

    def __post_init__(self) -> None:
        _finite_tuple(self.position, 3, "position")
        _finite_tuple(self.velocity, 3, "velocity")
        if not math.isfinite(float(self.yaw)):
            raise ValueError("yaw must be finite")
        if self.health not in {"ok", "degraded", "lost"}:
            raise ValueError(f"invalid health: {self.health}")


@dataclass(frozen=True)
class ModelObservation:
    episode_id: str
    instruction: str
    timestamp_ns: int
    image_timestamp_ns: Mapping[str, int]
    rgb_jpeg: Mapping[str, bytes]
    states: Mapping[str, VehicleState]
    action_history: Sequence[Mapping[str, Sequence[float]]] = field(default_factory=tuple)
    message_history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.instruction:
            raise ValueError("episode_id and instruction are required")
        if int(self.timestamp_ns) < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if set(self.rgb_jpeg) != set(AGENT_IDS):
            raise ValueError("rgb_jpeg must contain exactly uav_0 and uav_1")
        if set(self.states) != set(AGENT_IDS):
            raise ValueError("states must contain exactly uav_0 and uav_1")
        if set(self.image_timestamp_ns) != set(AGENT_IDS):
            raise ValueError("image timestamps must contain exactly uav_0 and uav_1")
        if any(int(self.image_timestamp_ns[key]) < 0 for key in AGENT_IDS):
            raise ValueError("image timestamps must be non-negative")
        if any(not isinstance(self.rgb_jpeg[key], bytes) or not self.rgb_jpeg[key] for key in AGENT_IDS):
            raise ValueError("each RGB observation must be non-empty JPEG bytes")
        skew = abs(int(self.image_timestamp_ns["uav_0"]) - int(self.image_timestamp_ns["uav_1"]))
        if skew > 100_000_000:
            raise ValueError(f"camera synchronization error is {skew / 1e6:.1f} ms (>100 ms)")
        action_skew = abs(int(self.timestamp_ns) - max(int(self.image_timestamp_ns[key]) for key in AGENT_IDS))
        if action_skew > 100_000_000:
            raise ValueError(f"image/action synchronization error is {action_skew / 1e6:.1f} ms (>100 ms)")

    def to_wire(self, *, include_images: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": MODEL_PROTOCOL_VERSION,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "timestamp_ns": int(self.timestamp_ns),
            "image_timestamp_ns": {key: int(self.image_timestamp_ns[key]) for key in AGENT_IDS},
            "states": {key: asdict(self.states[key]) for key in AGENT_IDS},
            "action_history": list(self.action_history)[-4:],
            "message_history": list(self.message_history)[-16:],
        }
        if include_images:
            payload["rgb_jpeg_b64"] = {
                key: base64.b64encode(self.rgb_jpeg[key]).decode("ascii") for key in AGENT_IDS
            }
        assert_truth_free(payload)
        return payload

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_wire(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")


@dataclass(frozen=True)
class WaypointDelta:
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    dyaw: float = 0.0

    def __post_init__(self) -> None:
        _finite_tuple((self.dx, self.dy, self.dz, self.dyaw), 4, "waypoint")

    def clipped(self) -> "WaypointDelta":
        return WaypointDelta(
            dx=max(-ACTION_LIMITS["dx"], min(ACTION_LIMITS["dx"], self.dx)),
            dy=max(-ACTION_LIMITS["dy"], min(ACTION_LIMITS["dy"], self.dy)),
            dz=max(-ACTION_LIMITS["dz"], min(ACTION_LIMITS["dz"], self.dz)),
            dyaw=max(-ACTION_LIMITS["dyaw"], min(ACTION_LIMITS["dyaw"], self.dyaw)),
        )

    def vector(self) -> list[float]:
        value = self.clipped()
        return [value.dx, value.dy, value.dz, value.dyaw]


@dataclass(frozen=True)
class ModelJointAction:
    uav_0: WaypointDelta = field(default_factory=WaypointDelta)
    uav_1: WaypointDelta = field(default_factory=WaypointDelta)
    intent: str = "none"

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ModelJointAction":
        if payload.get("protocol") not in {None, MODEL_PROTOCOL_VERSION}:
            raise ValueError("policy protocol version mismatch")
        agents = payload.get("waypoints", payload)
        return cls(
            uav_0=WaypointDelta(*_finite_tuple(agents["uav_0"], 4, "uav_0 waypoint")),
            uav_1=WaypointDelta(*_finite_tuple(agents["uav_1"], 4, "uav_1 waypoint")),
            intent=str(payload.get("intent", "none")),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol": MODEL_PROTOCOL_VERSION,
            "waypoints": {"uav_0": self.uav_0.vector(), "uav_1": self.uav_1.vector()},
            "intent": self.intent,
        }

    def to_velocity_action(self, period_s: float = 0.5) -> JointAction:
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("period_s must be positive and finite")
        actions: dict[str, Action] = {}
        for agent_id in AGENT_IDS:
            waypoint = getattr(self, agent_id).clipped()
            actions[agent_id] = Action(
                vx=waypoint.dx / period_s,
                vy=waypoint.dy / period_s,
                vz=waypoint.dz / period_s,
                yaw_rate=waypoint.dyaw / period_s,
            ).clipped()
        message = None
        if self.intent == "handoff":
            message = {"sender": "uav_0", "receiver": "uav_1", "payload": {"intent": "handoff_request"}}
        elif self.intent == "confirm":
            message = {"sender": "uav_1", "receiver": "uav_0", "payload": {"intent": "confirm"}}
        return JointAction(actions["uav_0"], actions["uav_1"], message)


def hover_action() -> ModelJointAction:
    return ModelJointAction()
