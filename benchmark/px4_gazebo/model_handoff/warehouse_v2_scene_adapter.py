"""Bridge live warehouse observations into the Warehouse v2 core contract.

The adapter is deliberately narrow: it accepts the existing truth-free live
``ModelObservation`` packet and emits a v2 ``ModelInput``.  It does not read
Gazebo truth, choose a source/receiver role, or claim that a disarmed capture
is a model score or flight result.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Mapping

from .core.model_io import ModelInput
from .protocol import AGENT_IDS, ModelObservation, assert_truth_free
from .language_tasks import TargetReport
from .warehouse_v2_task import HORIZONS_S, SCENE, TASK, _instruction


ADAPTER_ID = "warehouse_v2_live_scene_adapter_v2"


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
    stamps = [observation.timestamp_ns, *observation.image_timestamp_ns.values()]
    if any(type(stamp) is not int or stamp < 0 for stamp in stamps):
        raise ValueError("nonnegative integer capture timestamps required")
    if any(stamp > observation.timestamp_ns for stamp in observation.image_timestamp_ns.values()):
        raise ValueError("future camera exposure cannot enter current ModelInput")
    # Validate direct dataclass inputs as strictly as TCP packets. All history
    # is already delivered by the trusted runtime; timestamps alone cannot
    # certify network receipt, so this adapter grants no audit/score flags.
    ModelObservation.from_wire(wire)
    for message in observation.message_history:
        report = TargetReport.from_wire(message)
        if report.image_timestamp_ns >= min(observation.image_timestamp_ns.values()):
            raise ValueError("peer claim must precede both current exposures")
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
        if set(task_context)-{'role_assignment', 'role_mode', 'prediction_horizons_s', 'message_budget'}:
            raise ValueError("unknown or private task context field")
        if task_context.get("role_assignment") is not None or task_context.get('role_mode', 'unassigned') != 'unassigned':
            raise ValueError("roles must remain unassigned")
        horizons = task_context.get('prediction_horizons_s', list(HORIZONS_S))
        if list(horizons) != list(HORIZONS_S):
            raise ValueError("task prediction horizons cannot be overridden")
        budget = task_context.get('message_budget', 8)
        if type(budget) is not int or not 0 <= budget <= 8:
            raise ValueError("integer message budget in 0..8 required")
        context['message_budget'] = budget
    observations = {}
    for agent in AGENT_IDS:
        observations[agent] = {
            "rgb_jpeg_b64": wire["rgb_jpeg_b64"][agent],
            "local_state": deepcopy(wire['states'][agent]),
            "image_timestamp_ns": int(observation.image_timestamp_ns[agent]),
            "peer_messages": deepcopy(wire['message_history']),
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
        messages=tuple(deepcopy(wire['message_history'])),
        action_history=tuple(deepcopy(wire['action_history'])),
        task_context=context,
    )


def from_wire_to_model_input(payload: Mapping[str, Any], *, task_context: Mapping[str, Any] | None = None) -> ModelInput:
    """Decode one live TCP request and convert it without widening the input."""
    return to_model_input(ModelObservation.from_wire(payload), task_context=task_context)


def decode_rgb(input_item: ModelInput) -> dict[str, bytes]:
    """Decode only the public RGB fields for a local model adapter."""
    result = {}
    for agent in input_item.agent_ids:
        value = input_item.observations[agent].get("rgb_jpeg_b64")
        if not isinstance(value, str):
            raise ValueError("ModelInput is missing public rgb_jpeg_b64")
        result[agent] = base64.b64decode(value, validate=True)
    return result

