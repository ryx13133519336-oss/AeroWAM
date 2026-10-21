"""Bridge live warehouse observations into the Warehouse v2 core contract.

The adapter is deliberately narrow: it accepts the existing truth-free live
``ModelObservation`` packet and emits a v2 ``ModelInput``.  It does not read
Gazebo truth, choose a source/receiver role, or claim that a disarmed capture
is a model score or flight result.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any, Mapping

from .core.model_io import ModelInput
from .protocol import AGENT_IDS, ModelObservation, assert_truth_free
from .warehouse_v2_task import HORIZONS_S, SCENE, TASK, _instruction


ADAPTER_ID = "warehouse_v2_live_scene_adapter_v1"


def to_model_input(observation: ModelObservation, *, task_context: Mapping[str, Any] | None = None) -> ModelInput:
    """Convert one actual public capture to the v2 model input.

    Images are carried as base64 strings because the core wire format is JSON.
    Vehicle state is each vehicle's own telemetry only.  The adapter never
    derives role labels from the vehicle ID; role is inferred by the model and
    evaluated from private scene evidence later.
    """
    if not isinstance(observation, ModelObservation):
        raise TypeError("a validated ModelObservation is required")
    wire = observation.to_wire(include_images=True)
    assert_truth_free(wire)
    context = {
        "role_mode": "unassigned",
        "prediction_horizons_s": list(HORIZONS_S),
        "message_budget": 8,
        "observation_semantics": "live_dual_rgb_and_own_telemetry",
        "teammate_messages_are_claims": True,
        "adapter_id": ADAPTER_ID,
    }
    if task_context is not None:
        assert_truth_free(task_context, "task_context")
        if task_context.get("role_assignment") is not None:
            raise ValueError("role_assignment must remain null")
        context.update(dict(task_context))
    context["role_mode"] = "unassigned"
    context.pop("role_assignment", None)
    observations = {}
    for agent in AGENT_IDS:
        observations[agent] = {
            "rgb_jpeg_b64": wire["rgb_jpeg_b64"][agent],
            "local_state": asdict(observation.states[agent]),
            "image_timestamp_ns": int(observation.image_timestamp_ns[agent]),
            "peer_messages": list(observation.message_history),
        }
    return ModelInput(
        episode_id=observation.episode_id,
        task_id=TASK.task_id,
        task_version=TASK.task_version,
        scene_id=SCENE.scene_id,
        scene_family_id=SCENE.scene_family_id,
        instruction=_instruction() if not observation.instruction.strip() else observation.instruction,
        timestamp_ns=int(observation.timestamp_ns),
        agent_ids=tuple(AGENT_IDS),
        observations=observations,
        messages=tuple(observation.message_history),
        action_history=tuple(observation.action_history),
        task_context=context,
    )


def decode_rgb(input_item: ModelInput) -> dict[str, bytes]:
    """Decode only the public RGB fields for a local model adapter."""
    result = {}
    for agent in input_item.agent_ids:
        value = input_item.observations[agent].get("rgb_jpeg_b64")
        if not isinstance(value, str):
            raise ValueError("ModelInput is missing public rgb_jpeg_b64")
        result[agent] = base64.b64decode(value, validate=True)
    return result

