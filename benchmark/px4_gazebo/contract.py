"""Policy-facing contracts. ROS/Gazebo truth must not cross this boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

AGENT_IDS = ("uav_0", "uav_1")


def _finite(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name} must have {size} values")
    result = tuple(float(item) for item in value)
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class Action:
    """Body-FLU velocity/yaw-rate command in SI units."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    def clipped(self) -> "Action":
        return Action(max(-3.0, min(3.0, self.vx)), max(-3.0, min(3.0, self.vy)), max(-1.5, min(1.5, self.vz)), max(-1.0, min(1.0, self.yaw_rate)))

    def as_vector(self) -> tuple[float, float, float, float]:
        return _finite((self.vx, self.vy, self.vz, self.yaw_rate), 4, "action")


@dataclass(frozen=True)
class AgentObservation:
    agent_id: str
    rgb_uri: str
    own_position: tuple[float, float, float]
    own_velocity: tuple[float, float, float]
    own_yaw: float
    health: str = "ok"
    last_message: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.agent_id not in AGENT_IDS:
            raise ValueError(f"unknown agent id: {self.agent_id}")
        _finite(self.own_position, 3, "own_position")
        _finite(self.own_velocity, 3, "own_velocity")
        if self.health not in {"ok", "degraded", "lost"}:
            raise ValueError(f"invalid health: {self.health}")

    def to_policy_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "rgb_uri": self.rgb_uri, "own_state": {"position": list(self.own_position), "velocity": list(self.own_velocity), "yaw": self.own_yaw, "health": self.health}, "last_message": dict(self.last_message) if self.last_message else None}


@dataclass(frozen=True)
class JointAction:
    uav_0: Action = field(default_factory=Action)
    uav_1: Action = field(default_factory=Action)
    message: Mapping[str, Any] | None = None

    def clipped(self) -> "JointAction":
        return JointAction(self.uav_0.clipped(), self.uav_1.clipped(), self.message)


@dataclass(frozen=True)
class HandoffTask:
    task_id: str = "handoff_safe_occlusion_v3"
    template: str = "safe_route_static_occluder_v3"
    split: str = "target_zero_shot"
    instruction: str = (
        "UAV0 must acquire and track the marked target before the wall. "
        "UAV1 must route around the wall end, take over through occlusion, "
        "and verify the same target while UAV0 holds the support position."
    )
    target_identity: str = "target_0"
    horizon: int = 240
    control_hz: float = 10.0
    target_start: tuple[float, float, float] = (-2.0, -1.6, 3.0)
    target_end: tuple[float, float, float] = (8.0, -1.6, 3.0)
    # The wall separates UAV0's observation lane from the target path without
    # intersecting the target's physical body or the reference flight routes.
    occluder_center: tuple[float, float, float] = (2.2, -0.72, 3.0)
    occluder_half_extents: tuple[float, float, float] = (0.6, 0.45, 2.0)
    visibility_range: float = 15.0
    initial_positions: Mapping[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "uav_0": (0.0, 0.0, 3.0),
            "uav_1": (3.0, 0.35, 3.0),
        }
    )
    staging_position: tuple[float, float, float] = (3.9, -3.1, 3.0)
    staging_tolerance: float = 0.75
    collision_distance: float = 1.5
    interruption_limit_steps: int = 10
    required_identity_frames: int = 3
    handoff_request_lead_steps: int = 20
    post_handoff_tracking_steps: int = 30

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_start"] = list(self.target_start)
        result["target_end"] = list(self.target_end)
        result["occluder_center"] = list(self.occluder_center)
        result["occluder_half_extents"] = list(self.occluder_half_extents)
        result["staging_position"] = list(self.staging_position)
        result["initial_positions"] = {key: list(value) for key, value in self.initial_positions.items()}
        return result
