"""Bridge a legacy grounded policy into the SubT Core model protocol.

The bridge preserves per-camera evidence and dynamic event ownership. Legacy
body-FLU motion is held by default because it is not a verified SubT ENU
action frame. Proxy ENU motion is an explicit diagnostic opt-in only.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Callable, Mapping

from .core.model_io import MODEL_INPUT_SCHEMA, ModelOutput
from .grounded_interface import GroundedAction, validate_observed_reports
from .protocol import ModelObservation, VehicleState, assert_truth_free
from .subt_handoff_model_io import validate_output


EVENT_MAP = {
    "found": "handoff.found",
    "request": "handoff.handoff_request",
    "handoff_request": "handoff.handoff_request",
    "accept": "handoff.handoff_accept",
    "handoff_accept": "handoff.handoff_accept",
    "confirm": "handoff.confirm",
}
AGENTS = ("uav_0", "uav_1")


def _yaw(quaternion_xyzw: Mapping[str, Any] | list[Any] | tuple[Any, ...]) -> float:
    if len(quaternion_xyzw) != 4:
        raise ValueError("quaternion_xyzw must contain four values")
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise ValueError("quaternion contains a non-finite value")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _history_row(row: Mapping[str, Any]) -> dict[str, list[float]]:
    actions = row.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(AGENTS):
        raise ValueError("action history must contain both agents")
    result = {}
    for agent in AGENTS:
        action = actions[agent]
        if not isinstance(action, Mapping) or action.get("type") != "waypoint_delta":
            raise ValueError("unsupported action history entry")
        delta = action.get("delta_enu_m")
        yaw = action.get("delta_yaw_rad")
        if not isinstance(delta, list) or len(delta) != 3:
            raise ValueError("action history displacement must contain three values")
        values = [float(value) for value in (*delta, yaw)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action history contains a non-finite value")
        result[agent] = values
    return result


def core_to_legacy(payload: Mapping[str, Any]) -> ModelObservation:
    """Convert one public ``aerowam-model-input/1`` request to the old view."""
    assert_truth_free(payload)
    if payload.get("schema") != MODEL_INPUT_SCHEMA:
        raise ValueError("SubT Core model input schema required")
    if set(payload.get("agent_ids", ())) != set(AGENTS):
        raise ValueError("SubT bridge requires exactly two agents")
    observations = payload.get("observations")
    if not isinstance(observations, Mapping) or set(observations) != set(AGENTS):
        raise ValueError("SubT Core observations must contain both agents")
    rgb, stamps, states = {}, {}, {}
    for agent in AGENTS:
        row = observations[agent]
        if not isinstance(row, Mapping):
            raise ValueError("agent observation must be an object")
        encoded = row.get("rgb_jpeg_b64")
        stamp = row.get("image_timestamp_ns", payload["timestamp_ns"])
        proprio = row.get("proprioception")
        if not isinstance(encoded, str) or not isinstance(stamp, int) or not isinstance(proprio, Mapping):
            raise ValueError("RGB, timestamp and proprioception are required")
        try:
            rgb[agent] = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("Invalid RGB base64") from exc
        position = tuple(float(value) for value in proprio["position_local_m"])
        quaternion = tuple(float(value) for value in proprio["quaternion_xyzw"])
        if len(position) != 3 or len(quaternion) != 4:
            raise ValueError("Invalid proprioception dimensions")
        stamps[agent] = stamp
        states[agent] = VehicleState(position, (0.0, 0.0, 0.0), _yaw(quaternion), "ok")
    history = tuple(_history_row(row) for row in payload.get("action_history", ()))
    return ModelObservation(
        episode_id=payload["episode_id"], instruction=payload["instruction"],
        timestamp_ns=payload["timestamp_ns"], image_timestamp_ns=stamps,
        rgb_jpeg=rgb, states=states, action_history=history,
        message_history=tuple(payload.get("messages", ())),
    )


def grounded_to_core(payload: Mapping[str, Any], result: GroundedAction,
                     *, allow_proxy_enu_actions: bool = False) -> dict[str, Any]:
    """Convert grounded evidence while making the action-frame decision explicit."""
    if not isinstance(result, GroundedAction):
        raise TypeError("grounded policy must return GroundedAction")
    observation = core_to_legacy(payload)
    validate_observed_reports(result, observation)
    actions = {}
    for agent in AGENTS:
        if allow_proxy_enu_actions:
            waypoint = getattr(result.action, agent)
            actions[agent] = {"type": "waypoint_delta", "delta_enu_m": [
                float(waypoint.dx), float(waypoint.dy), float(waypoint.dz)],
                "delta_yaw_rad": float(waypoint.dyaw)}
        else:
            actions[agent] = {"type": "waypoint_delta", "delta_enu_m": [0.0, 0.0, 0.0],
                              "delta_yaw_rad": 0.0}
    events = []
    for report in result.reports:
        event = EVENT_MAP.get(report.event)
        if event is None:
            continue
        events.append({"agent_id": report.agent_id, "event": event,
                       "image_timestamp_ns": report.image_timestamp_ns,
                       "bbox_xyxy": list(report.bbox_xyxy)})
    output = ModelOutput(episode_id=payload["episode_id"], timestamp_ns=payload["timestamp_ns"],
                         actions=actions, events=tuple(events), predictions={}).to_wire()
    validate_output(output, payload)
    return output


def make_core_policy(grounded_policy: Callable[[ModelObservation], GroundedAction], *,
                     allow_proxy_enu_actions: bool = False) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Wrap a legacy grounded callable for ``subt_handoff_proxy``."""
    if not callable(grounded_policy):
        raise TypeError("grounded_policy must be callable")

    def policy(payload: Mapping[str, Any]) -> dict[str, Any]:
        observation = core_to_legacy(payload)
        result = grounded_policy(observation)
        return grounded_to_core(payload, result, allow_proxy_enu_actions=allow_proxy_enu_actions)

    return policy
