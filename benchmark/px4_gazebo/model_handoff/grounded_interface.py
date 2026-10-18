"""Opt-in action/evidence envelope and adapter. Live flight runner pending."""

from dataclasses import dataclass
from typing import Mapping

from .language_tasks import TargetReport, exact_keys
from .protocol import AGENT_IDS, ModelJointAction, ModelObservation, assert_truth_free

GROUNDED_ACTION_SCHEMA = "aerowam-grounded-action/1-draft"


@dataclass(frozen=True)
class GroundedAction:
    action: ModelJointAction
    reports: tuple[TargetReport, ...] = ()

    def __post_init__(self):
        if not isinstance(self.action, ModelJointAction):
            raise ValueError("ModelJointAction required")
        if not isinstance(self.reports, tuple) or any(not isinstance(r, TargetReport) for r in self.reports):
            raise ValueError("Immutable target reports required")
        if len(self.reports) > len(AGENT_IDS) or len({r.agent_id for r in self.reports}) != len(self.reports):
            raise ValueError("At most one report per UAV per response")

    def to_wire(self) -> dict:
        result = {"schema": GROUNDED_ACTION_SCHEMA, "action": self.action.to_wire(),
                  "reports": [r.to_wire() for r in self.reports]}
        assert_truth_free(result)
        return result

    @classmethod
    def from_wire(cls, payload: Mapping) -> "GroundedAction":
        assert_truth_free(payload)
        exact_keys(payload, {"schema", "action", "reports"})
        if payload["schema"] != GROUNDED_ACTION_SCHEMA or not isinstance(payload["reports"], list):
            raise ValueError("Grounded action schema/reports mismatch")
        exact_keys(payload["action"], {"protocol", "waypoints", "intent"})
        exact_keys(payload["action"]["waypoints"], set(AGENT_IDS))
        return cls(ModelJointAction.from_wire(payload["action"]),
                   tuple(TargetReport.from_wire(row) for row in payload["reports"]))


def validate_observed_reports(result: GroundedAction, observation: ModelObservation) -> None:
    """Reports reference only the response's public camera stamps.

    No private labels are needed here. Identity/visibility acceptance remains
    the evaluator's job; matching a stamp alone never proves task success.
    """
    for report in result.reports:
        if report.image_timestamp_ns != observation.image_timestamp_ns[report.agent_id]:
            raise ValueError("Target report must reference this request's observed frame")


class GroundedAdapter:
    """Adapter-server callable: policy receives validated public observation.

    The supplied policy maps ModelObservation to GroundedAction. No task
    binding, private frame annotation or evaluator result is passed to it.
    This is the centralized diagnostic interface, not restricted messaging.
    """

    def __init__(self, policy):
        if not callable(policy):
            raise TypeError("Grounded policy must be callable")
        self.policy = policy

    def __call__(self, payload: Mapping) -> dict:
        observation = ModelObservation.from_wire(payload)
        result = self.policy(observation)
        if not isinstance(result, GroundedAction):
            raise TypeError("Grounded policy must return GroundedAction")
        validate_observed_reports(result, observation)
        return result.to_wire()
