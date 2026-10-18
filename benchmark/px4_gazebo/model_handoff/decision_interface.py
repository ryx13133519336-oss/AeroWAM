"""Draft v1 candidate-conditioned interface; no evaluator or flight executor.

The centralized request contains explicit observed history and the unchanged
language instruction. Candidate actions are body-FLU waypoint deltas over each
step's period, not oracle routes, absolute scene positions or executed controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Protocol, Sequence

from .protocol import AGENT_IDS, ModelJointAction, ModelObservation, assert_truth_free

DECISION_PROTOCOL_VERSION = "aerowam-candidate-decision/1-draft"


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _probability(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")


def _keys(payload: Mapping, expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError(f"Interface fields differ: expected {sorted(expected)}")


@dataclass(frozen=True)
class CandidateStep:
    action: ModelJointAction
    duration_s: float

    def __post_init__(self) -> None:
        _positive(self.duration_s, "duration_s")
        # Never silently clip a candidate while labeling it as the original.
        for agent in AGENT_IDS:
            delta = getattr(self.action, agent)
            if [delta.dx, delta.dy, delta.dz, delta.dyaw] != delta.vector():
                raise ValueError("Candidate exceeds waypoint limits")
            velocity = getattr(self.action.to_velocity_action(self.duration_s), agent)
            if any(not math.isclose(v, d / self.duration_s, abs_tol=1e-12)
                   for v, d in zip(velocity.as_vector(), delta.vector())):
                raise ValueError("Candidate duration would clip its velocity command")

    def to_wire(self) -> dict:
        action = self.action.to_wire()
        action["waypoints"] = {agent: [float(v) for v in action["waypoints"][agent]] for agent in AGENT_IDS}
        return {"action": action, "duration_s": float(self.duration_s)}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    steps: tuple[CandidateStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id or not self.steps:
            raise ValueError("Candidate ID and action sequence are required")

    def to_wire(self) -> dict:
        return {"candidate_id": self.candidate_id, "steps": [step.to_wire() for step in self.steps]}

    @classmethod
    def from_wire(cls, payload: Mapping) -> "Candidate":
        _keys(payload, {"candidate_id", "steps"})
        steps = []
        for step in payload["steps"]:
            _keys(step, {"action", "duration_s"})
            _keys(step["action"], {"protocol", "waypoints", "intent"})
            _keys(step["action"]["waypoints"], set(AGENT_IDS))
            steps.append(CandidateStep(ModelJointAction.from_wire(step["action"]), step["duration_s"]))
        return cls(payload["candidate_id"], tuple(steps))


@dataclass(frozen=True)
class DecisionRequest:
    decision_id: str
    task: str
    instruction: str
    history: tuple[ModelObservation, ...]
    candidates: tuple[Candidate, ...]
    horizon_s: float

    def __post_init__(self) -> None:
        if not self.decision_id or self.task not in {"handoff", "search", "inspection"} or not self.instruction:
            raise ValueError("Decision ID, supported task and language instruction are required")
        _positive(self.horizon_s, "horizon_s")
        if not self.history or not self.candidates:
            raise ValueError("Observed history and candidates are required")
        episode = self.history[0].episode_id
        previous = -1
        for observation in self.history:
            if observation.episode_id != episode or observation.instruction != self.instruction:
                raise ValueError("Mixed episode/instruction history")
            if observation.timestamp_ns <= previous:
                raise ValueError("Non-increasing observation history")
            previous = observation.timestamp_ns
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate candidate IDs")
        for candidate in self.candidates:
            if not math.isclose(sum(step.duration_s for step in candidate.steps), self.horizon_s, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Candidates must share the declared prediction horizon")

    def to_wire(self) -> dict:
        result = {
            "protocol": DECISION_PROTOCOL_VERSION, "decision_id": self.decision_id,
            "task": self.task, "instruction": self.instruction, "horizon_s": self.horizon_s,
            "history": [item.to_wire() for item in self.history],
            "candidates": [candidate.to_wire() for candidate in self.candidates],
        }
        assert_truth_free(result)
        return result

    @classmethod
    def from_wire(cls, payload: Mapping) -> "DecisionRequest":
        assert_truth_free(payload)
        _keys(payload, {"protocol", "decision_id", "task", "instruction", "horizon_s", "history", "candidates"})
        if payload["protocol"] != DECISION_PROTOCOL_VERSION:
            raise ValueError("Decision protocol mismatch")
        return cls(
            payload["decision_id"], payload["task"], payload["instruction"],
            tuple(ModelObservation.from_wire(row) for row in payload["history"]),
            tuple(Candidate.from_wire(row) for row in payload["candidates"]), payload["horizon_s"],
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CandidatePrediction:
    candidate_id: str
    utility: float
    target_visibility_probability: Mapping[str, float]
    handoff_success_probability: float
    confirmation_probability: float
    collision_probability: float

    def __post_init__(self) -> None:
        if not self.candidate_id or isinstance(self.utility, bool) or not math.isfinite(self.utility):
            raise ValueError("Invalid candidate ID/utility")
        if set(self.target_visibility_probability) != set(AGENT_IDS):
            raise ValueError("Visibility probabilities require both vehicles")
        for value in (*self.target_visibility_probability.values(), self.handoff_success_probability,
                      self.confirmation_probability, self.collision_probability):
            _probability(value)

    def to_wire(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "utility": self.utility,
            "target_visibility_probability": dict(self.target_visibility_probability),
            "handoff_success_probability": self.handoff_success_probability,
            "confirmation_probability": self.confirmation_probability,
            "collision_probability": self.collision_probability,
        }

    @classmethod
    def from_wire(cls, payload: Mapping) -> "CandidatePrediction":
        _keys(payload, {"candidate_id", "utility", "target_visibility_probability",
                       "handoff_success_probability", "confirmation_probability", "collision_probability"})
        return cls(**payload)


def validate_predictions(request: DecisionRequest, predictions: Sequence[CandidatePrediction]) -> None:
    identifiers = [row.candidate_id for row in predictions]
    if len(identifiers) != len(request.candidates) or set(identifiers) != {row.candidate_id for row in request.candidates}:
        raise ValueError("Predictions must cover each candidate exactly once")
    for prediction in predictions:
        # Revalidate mutable nested mappings before selection/serialization.
        CandidatePrediction.from_wire(prediction.to_wire())


def select(request: DecisionRequest, predictions: Sequence[CandidatePrediction], *, max_collision_probability: float | None = None) -> str | None:
    """Fixed utility selector with optional risk filtering; None means abstain.

    Probabilities are model estimates, not certified safety guarantees. Utility
    definition and threshold must be frozen by the experiment, not this module.
    Stable ties follow request order, independent of response order.
    """
    validate_predictions(request, predictions)
    if max_collision_probability is not None:
        _probability(max_collision_probability)
    by_id = {row.candidate_id: row for row in predictions}
    eligible = [by_id[row.candidate_id] for row in request.candidates
                if max_collision_probability is None or by_id[row.candidate_id].collision_probability <= max_collision_probability]
    return max(eligible, key=lambda row: row.utility).candidate_id if eligible else None


class DecisionPredictor(Protocol):
    def predict(self, *, history: Sequence[ModelObservation], instruction: str,
                candidates: Sequence[Candidate], horizon_s: float, task: str) -> Sequence[CandidatePrediction]: ...


class DecisionAdapter:
    """Serve an external predictor through the common length-prefixed transport."""

    def __init__(self, predictor: DecisionPredictor) -> None:
        self.predictor = predictor

    def __call__(self, payload: Mapping[str, Any]) -> dict:
        request = DecisionRequest.from_wire(payload)
        predictions = tuple(self.predictor.predict(
            history=request.history, instruction=request.instruction, candidates=request.candidates,
            horizon_s=request.horizon_s, task=request.task,
        ))
        validate_predictions(request, predictions)
        return {"protocol": DECISION_PROTOCOL_VERSION, "decision_id": request.decision_id,
                "predictions": [row.to_wire() for row in predictions]}


class DecisionClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 0.4) -> None:
        from .policy_client import PolicyClient
        self.transport = PolicyClient(endpoint, timeout_s=timeout_s)

    def predict(self, request: DecisionRequest) -> tuple[tuple[CandidatePrediction, ...], float]:
        response, latency_ms = self.transport.request(request.canonical_bytes())
        _keys(response, {"protocol", "decision_id", "predictions"})
        if response["protocol"] != DECISION_PROTOCOL_VERSION or response["decision_id"] != request.decision_id:
            raise ValueError("Mismatched decision response")
        predictions = tuple(CandidatePrediction.from_wire(row) for row in response["predictions"])
        validate_predictions(request, predictions)
        return predictions, latency_ms
