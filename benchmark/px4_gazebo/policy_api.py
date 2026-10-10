"""Stable policy-facing API for VLN, WAM, and rule-based backends.

The objects in this module deliberately contain observations only. Simulator
truth belongs in the evaluator-side episode record and must not be copied into
these inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Protocol, Sequence

from .contract import Action


def _vector(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class Prediction:
    """Optional world-model output, expressed in the policy's local frame."""

    horizon_steps: int
    target_positions: tuple[tuple[float, float, float], ...] = ()
    teammate_positions: tuple[tuple[float, float, float], ...] = ()
    target_visible_probability: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_steps < 0:
            raise ValueError("horizon_steps must be non-negative")
        for name, sequence in (
            ("target_positions", self.target_positions),
            ("teammate_positions", self.teammate_positions),
        ):
            if len(sequence) > self.horizon_steps:
                raise ValueError(f"{name} is longer than horizon_steps")
            for point in sequence:
                _vector(point, 3, name)
        if len(self.target_visible_probability) > self.horizon_steps:
            raise ValueError("target_visible_probability is longer than horizon_steps")
        if not all(0.0 <= float(value) <= 1.0 for value in self.target_visible_probability):
            raise ValueError("target visibility probabilities must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_steps": self.horizon_steps,
            "target_positions": [list(point) for point in self.target_positions],
            "teammate_positions": [list(point) for point in self.teammate_positions],
            "target_visible_probability": list(self.target_visible_probability),
        }


@dataclass(frozen=True)
class PolicyOutput:
    """One synchronized policy decision for one UAV."""

    action: Action = field(default_factory=Action)
    message: Mapping[str, Any] | None = None
    prediction: Prediction | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.action.as_vector()
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.message is not None:
            try:
                json.dumps(self.message, ensure_ascii=True, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError("message must be JSON serializable") from exc

    def clipped(self) -> "PolicyOutput":
        return PolicyOutput(
            action=self.action.clipped(),
            message=self.message,
            prediction=self.prediction,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": {"control": list(self.action.as_vector())},
            "message": dict(self.message) if self.message is not None else None,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PolicyContext:
    """The complete information a backend may use at one decision point."""

    task: Mapping[str, Any]
    agent_id: str
    observation: Mapping[str, Any]
    timestamp_ms: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": dict(self.task),
            "agent_id": self.agent_id,
            "observation": dict(self.observation),
            "timestamp_ms": self.timestamp_ms,
            "seed": self.seed,
        }


class PolicyBackend(Protocol):
    """Adapter contract implemented by every policy family."""

    def reset(self, task_spec: Mapping[str, Any], agent_id: str, seed: int) -> None:
        """Start a new episode for one agent."""

    def act(self, context: PolicyContext) -> PolicyOutput:
        """Return an action using only the supplied policy context."""

    def close(self) -> None:
        """Release model/server resources after an episode or batch."""

