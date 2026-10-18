"""Expert action labels; pending confirmation survives the 2 Hz recorder."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contract import JointAction
from .protocol import ModelJointAction, WaypointDelta


def waypoint_label(requested: JointAction, *, confirm: bool = False) -> ModelJointAction:
    waypoints = [
        WaypointDelta(action.vx * 0.5, action.vy * 0.5, action.vz * 0.5, action.yaw_rate * 0.5).clipped()
        for action in (requested.uav_0, requested.uav_1)
    ]
    intent = "confirm" if confirm else "none"
    if requested.message is not None:
        payload = requested.message.get("payload", requested.message)
        if payload.get("intent") == "handoff_request":
            intent = "handoff"
        elif payload.get("intent") == "confirm":
            intent = "confirm"
    return ModelJointAction(*waypoints, intent)


def confirmation_label(
    label: ModelJointAction, *, pending: bool, visible: bool,
) -> tuple[ModelJointAction, bool]:
    """Consume confirmation only on a visible recorded row, after verification.

    Keep a simultaneous handoff request intact; it must remain supervised.
    This is an expert annotation, not a claim that a confirm packet was sent.
    """
    if pending and visible and label.intent != "handoff":
        return ModelJointAction(label.uav_0, label.uav_1, "confirm"), False
    return label, pending


def has_verification(events: Sequence[Mapping[str, Any]]) -> bool:
    return any(event.get("event_type") == "verify" and event.get("success") for event in events)
