"""SubT binding of the published multi-task core; no new model protocol.

Camera-proxy state remains labelled as such. No velocity, health, flight
telemetry, hidden route or current teacher events are fabricated for a model.
"""

import base64
from collections.abc import Mapping
import math

from .core.model_io import ModelInput, ModelOutput, MODEL_OUTPUT_SCHEMA
from .core._common import assert_public, canonical
from .protocol import ACTION_LIMITS
from .subt_handoff_inspect import safe_image
from .subt_handoff_route import AGENTS

EVENTS = {"handoff.found", "handoff.handoff_request", "handoff.handoff_accept", "handoff.confirm"}


def model_input(root, row, *, episode_id, instruction, messages=(), history=()):
    observations = {}
    for agent in AGENTS:
        observations[agent] = {
            "rgb_jpeg_b64": base64.b64encode(safe_image(root, row["rgb"][agent]).read_bytes()).decode("ascii"),
            "image_timestamp_ns": row["stamp_ns"],
            "proprioception": row["proprioception"][agent],
            "state_source": "ideal_camera_proxy_not_px4",
        }
    return ModelInput(episode_id=episode_id, task_id="handoff", task_version="1",
        scene_id="subt_urban_stair_v1", scene_family_id="subt_urban_2_story",
        instruction=instruction, timestamp_ns=row["stamp_ns"], agent_ids=AGENTS,
        observations=observations, messages=tuple(messages), action_history=tuple(history[-4:]),
        task_context={"message_delay_s": .5, "dt_s": row["dt_s"],
                      "action_frame": "proxy_enu_displacement_m_yaw_rad", "role_assignment": None}).to_wire()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Finite numeric action/prediction required")
    return float(value)


def validate_output(payload, request):
    if not isinstance(payload, Mapping) or set(payload) != {
            "schema", "episode_id", "timestamp_ns", "actions", "events", "predictions"}:
        raise ValueError("Expected aerowam-model-output/1 exact fields")
    if (payload["schema"] != MODEL_OUTPUT_SCHEMA or payload["episode_id"] != request["episode_id"]
            or type(payload["timestamp_ns"]) is not int or payload["timestamp_ns"] != request["timestamp_ns"]):
        raise ValueError("Model response episode/time/schema mismatch")
    assert_public(payload)
    if set(payload["actions"]) != set(AGENTS):
        raise ValueError("Both per-agent actions are required")
    for action in payload["actions"].values():
        if set(action) != {"type", "delta_enu_m", "delta_yaw_rad"} or action["type"] != "waypoint_delta":
            raise ValueError("Expected explicit waypoint_delta in proxy ENU frame")
        delta = action["delta_enu_m"]
        if not isinstance(delta, list) or len(delta) != 3:
            raise ValueError("Three ENU displacement components required")
        for value, bound in zip(delta + [action["delta_yaw_rad"]], ACTION_LIMITS.values()):
            if abs(number(value)) > bound:
                raise ValueError("Action exceeds published limits; silent clipping refused")
    if not isinstance(payload["events"], list) or len(payload["events"]) > 2:
        raise ValueError("At most one grounded event per agent per decision")
    seen = set()
    for event in payload["events"]:
        if set(event) != {"agent_id", "event", "image_timestamp_ns", "bbox_xyxy"}:
            raise ValueError("Grounded task event fields invalid")
        agent = event["agent_id"]
        if agent not in AGENTS or agent in seen or event["event"] not in EVENTS:
            raise ValueError("Invalid or duplicate per-agent event")
        seen.add(agent)
        if (type(event["image_timestamp_ns"]) is not int
                or event["image_timestamp_ns"] != request["observations"][agent]["image_timestamp_ns"]):
            raise ValueError("Event must cite this decision's camera frame")
        box = event["bbox_xyxy"]
        if box is None and event["event"] == "handoff.handoff_request":
            continue  # Request is permitted after source loss of sight.
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Visual claim needs a normalized bounding box")
        x1,y1,x2,y2 = map(number, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid normalized box")
    ModelOutput(episode_id=payload["episode_id"], timestamp_ns=payload["timestamp_ns"],
        actions=payload["actions"], events=tuple(payload["events"]), predictions=payload["predictions"])
    canonical(payload)
    return dict(payload)


def hold(request):
    """Transport reference only; it makes no perception or task-success claim."""
    return ModelOutput(episode_id=request["episode_id"], timestamp_ns=request["timestamp_ns"],
        actions={a: {"type": "waypoint_delta", "delta_enu_m": [0.,0.,0.], "delta_yaw_rad": 0.}
                 for a in AGENTS}).to_wire()


def hold_factory():
    return hold
