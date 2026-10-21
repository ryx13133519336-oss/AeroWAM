"""Frozen SubT event-head adapter for the published core protocol.

This adapter deliberately abstains from grounded events because the current
checkpoint has no detector or bounding-box head. It is a protocol diagnostic:
the Ridge next-observation model is not used as an action-conditioned world
model, and zero actions are not task performance.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from io import BytesIO

from .core.model_io import ModelOutput
from .subt_handoff_learn import AGENTS, EVENTS, agent_features


def _training_event(value):
    return value.removeprefix("handoff.") if isinstance(value, str) else value


def _image_features(request):
    values = []
    for agent in AGENTS:
        raw = base64.b64decode(request["observations"][agent]["rgb_jpeg_b64"], validate=True)
        with Image.open(BytesIO(raw)) as frame:
            image = frame.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
            values.extend(np.asarray(image, dtype=np.float32).reshape(-1) / 255.)
    return values


def public_features(request):
    """Rebuild exactly the training public feature layout from core input."""
    image = _image_features(request)
    pose = []
    for agent in AGENTS:
        proprio = request["observations"][agent]["proprioception"]
        pose.extend(proprio["position_local_m"])
        pose.extend(proprio["quaternion_xyzw"])
    counts = np.zeros((len(AGENTS), len(EVENTS)), dtype=np.float32)
    recent = np.zeros_like(counts)
    messages = request.get("messages", [])
    for index, message in enumerate(messages):
        agent = message.get("agent_id")
        event = _training_event(message.get("event"))
        if agent in AGENTS and event in EVENTS:
            counts[AGENTS.index(agent), EVENTS.index(event)] += 1.
            recent[AGENTS.index(agent), EVENTS.index(event)] = index + 1
    recent /= max(1, len(messages))
    dt = request.get("task_context", {}).get("dt_s")
    return np.asarray(image + pose + counts.reshape(-1).tolist() + recent.reshape(-1).tolist()
                      + [0. if dt is None else float(dt)], dtype=np.float32)


def _hold_actions():
    return {agent: {"type": "waypoint_delta", "delta_enu_m": [0., 0., 0.],
                    "delta_yaw_rad": 0.} for agent in AGENTS}


class FrozenEventHead:
    def __init__(self, checkpoint):
        import joblib
        self.checkpoint = Path(checkpoint).resolve()
        self.policy = joblib.load(self.checkpoint / "event_policy.joblib")
        self.result = json.loads((self.checkpoint / "result.json").read_text(encoding="utf8"))
        if self.result.get("model_score_eligible") is not False:
            raise ValueError("Checkpoint must remain diagnostic-only")

    def __call__(self, request):
        if request.get("schema") != "aerowam-model-input/1":
            raise ValueError("Expected aerowam-model-input/1")
        features = public_features(request)
        predictions = np.asarray(self.policy.predict(agent_features(features.reshape(1, -1))))
        if predictions.shape != (len(AGENTS),):
            raise ValueError("Unexpected shared event-head output shape")
        # Do not emit an event without an exposure-aligned target box. The
        # prediction is retained as a diagnostic probability/label only.
        scores = {}
        if hasattr(self.policy, "predict_proba"):
            probabilities = np.asarray(self.policy.predict_proba(
                agent_features(features.reshape(1, -1))))
            classes = list(self.policy.classes_)
            for row, agent in enumerate(AGENTS):
                for index, label in enumerate(classes):
                    scores[f"event.{agent}.{label}"] = float(probabilities[row, index])
        for agent, label in zip(AGENTS, predictions.tolist()):
            scores[f"predicted_event.{agent}"] = float(EVENTS.index(label)) / max(1, len(EVENTS) - 1)
        return ModelOutput(episode_id=request["episode_id"], timestamp_ns=request["timestamp_ns"],
                           actions=_hold_actions(), events=tuple(), predictions=scores).to_wire()


def create_policy(checkpoint):
    """Factory used by the Windows TCP server."""
    return FrozenEventHead(checkpoint)
