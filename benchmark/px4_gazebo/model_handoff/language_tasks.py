"""Draft language-bound multi-task definitions; scene identities stay private."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .fixtures import Appearance, COLORS, SHAPES, STRIPES
from .protocol import AGENT_IDS, assert_truth_free

TASK_SCHEMA = "aerowam-language-task/1-draft"
ROLE_FREE_TASK_SCHEMA = "aerowam-language-task/2-draft"
GROUNDING_SCHEMA = "aerowam-target-report/1-draft"
TASKS = {"search", "handoff", "inspection"}
REPORT_EVENTS = {"observe", "found", "handoff_request", "handoff_accept", "confirm", "inspect"}


def exact_keys(value: Mapping, keys: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Unexpected or missing task/report fields")


def integer(value, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def finite_number(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def bbox(value) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("BBox must contain normalized x1,y1,x2,y2")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("Invalid bbox values")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError("BBox must be nonempty and within [0,1]")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class TaskRules:
    min_dwell_s: float = .5
    max_evidence_gap_s: float = .2
    min_iou: float = .5
    ambiguity_margin: float = .1
    max_report_age_s: float = .2
    inspection_views: tuple[str, ...] = ("front", "side")

    def __post_init__(self):
        for value in (self.min_dwell_s, self.max_evidence_gap_s, self.max_report_age_s):
            finite_number(value, "dwell/gap/report age")
            if value <= 0:
                raise ValueError("Positive finite dwell/gap required")
        finite_number(self.min_iou, "min_iou")
        finite_number(self.ambiguity_margin, "ambiguity_margin")
        if not 0 < self.min_iou <= 1 or not 0 < self.ambiguity_margin <= 1:
            raise ValueError("IoU and ambiguity thresholds must be in (0,1]")
        if not isinstance(self.inspection_views, tuple):
            raise ValueError("Inspection views must be an immutable tuple")
        if any(not isinstance(view, str) or not view.strip() for view in self.inspection_views):
            raise ValueError("Invalid inspection view")
        if len(self.inspection_views) < 2 or len(set(self.inspection_views)) != len(self.inspection_views):
            raise ValueError("Inspection requires at least two distinct views")


@dataclass(frozen=True)
class LanguageTask:
    """Author-selected descriptor, not an arbitrary-language parser."""
    task: str
    instruction: str
    descriptor: Appearance
    rules: TaskRules = TaskRules()
    source_agent: str | None = "uav_0"
    receiver_agent: str | None = "uav_1"
    # /1 defaults remain for legacy diagnostics, not the role-free benchmark.
    role_mode: str = "fixed"

    def __post_init__(self):
        if self.task not in TASKS or not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Supported task and instruction required")
        if not isinstance(self.descriptor, Appearance) or not isinstance(self.rules, TaskRules):
            raise ValueError("Appearance descriptor and TaskRules required")
        if (self.descriptor.color not in COLORS or self.descriptor.shape not in SHAPES
                or self.descriptor.stripe not in STRIPES):
            raise ValueError("Unsupported target appearance")
        if self.role_mode == "unassigned":
            if self.source_agent is not None or self.receiver_agent is not None:
                raise ValueError("Unassigned tasks cannot prescribe agent roles")
        elif self.role_mode == "fixed":
            if self.source_agent not in AGENT_IDS or self.receiver_agent not in AGENT_IDS or self.source_agent == self.receiver_agent:
                raise ValueError("Distinct valid task roles required")
        else:
            raise ValueError("Unsupported task role mode")

    def public_context(self) -> dict:
        # Do not expose the structured descriptor or evaluator answer ID.
        if self.role_mode == "unassigned":
            result = {"schema": ROLE_FREE_TASK_SCHEMA, "task": self.task, "instruction": self.instruction,
                      "role_mode": "unassigned", "rules": asdict(self.rules)}
        else:
            result = {"schema": TASK_SCHEMA, "task": self.task, "instruction": self.instruction,
                      "roles": {"source": self.source_agent, "receiver": self.receiver_agent},
                      "rules": asdict(self.rules)}
        assert_truth_free(result)
        return result

    def bind(self, objects: Mapping[str, Appearance]) -> "BoundTask":
        if not isinstance(objects, Mapping) or not objects or any(not isinstance(key, str) or not key for key in objects):
            raise ValueError("Private scene object identities required")
        if any(not isinstance(value, Appearance) for value in objects.values()):
            raise ValueError("Private scene appearances required")
        matching = [key for key, appearance in objects.items() if appearance == self.descriptor]
        if len(matching) != 1:
            raise ValueError("Instruction descriptor must identify exactly one scene object")
        return BoundTask(self, matching[0], tuple(objects))


@dataclass(frozen=True)
class BoundTask:
    definition: LanguageTask
    target_id: str
    object_ids: tuple[str, ...]

    def __post_init__(self):
        if (not isinstance(self.definition, LanguageTask) or not isinstance(self.object_ids, tuple)
                or any(not isinstance(key, str) or not key for key in self.object_ids)):
            raise ValueError("Invalid private scene binding")
        if self.target_id not in self.object_ids or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Invalid private target binding")


def role_free_task(task: str, descriptor: Appearance, *, instruction: str | None = None,
                   rules: TaskRules | None = None) -> LanguageTask:
    """New task entry point: agent identities do not prescribe responsibilities.

    Evidence may identify the source of one discovery/transfer, not a permanent
    scout/receiver role. A custom instruction remains human-authored and must
    be checked for consistency; this is not an arbitrary-language parser.
    """
    phrase = descriptor.phrase()
    instructions = {
        "search": f"Find the {phrase}. One UAV must report it; the other must independently confirm the same object.",
        "handoff": f"Track the {phrase}. Transfer tracking to the other UAV, which must track and confirm the same object.",
        "inspection": f"Both UAVs must inspect the {phrase} from complementary required views and confirm that object.",
    }
    if task not in instructions:
        raise ValueError("Unsupported task")
    return LanguageTask(task, instructions[task] if instruction is None else instruction, descriptor,
                        rules or TaskRules(), None, None, "unassigned")


def templated_task(task: str, descriptor: Appearance, *, rules: TaskRules | None = None) -> LanguageTask:
    """Legacy fixed-role diagnostic; new benchmark tasks use role_free_task."""
    phrase = descriptor.phrase()
    instructions = {
        "search": f"Find the {phrase}. UAV0 must report it; UAV1 must independently confirm the same object.",
        "handoff": f"UAV0 must track the {phrase}; hand it to UAV1, which must track and confirm the same object.",
        "inspection": f"Both UAVs must inspect the {phrase} from complementary required views and confirm that object.",
    }
    if task not in instructions:
        raise ValueError("Unsupported task")
    return LanguageTask(task, instructions[task], descriptor, rules or TaskRules())


def paired_target_tasks(task: str, objects: Mapping[str, Appearance], *, rules: TaskRules | None = None,
                        role_mode: str = "fixed") -> tuple[BoundTask, ...]:
    """Same private scene, one changed instruction/answer per distinct object.

    These conditions are a scene-level paired group, not independent frames.
    The live scene generator and grouped split scheduler are not implemented.
    """
    if len(objects) < 2:
        raise ValueError("A language-target pair requires at least two objects")
    if role_mode not in {"fixed", "unassigned"}:
        raise ValueError("Unsupported task role mode")
    factory = role_free_task if role_mode == "unassigned" else templated_task
    result = tuple(factory(task, appearance, rules=rules).bind(objects) for appearance in objects.values())
    if len({bound.target_id for bound in result}) != len(objects):
        raise ValueError("Target pairs must change the evaluator answer")
    return result


@dataclass(frozen=True)
class TargetReport:
    """Model evidence: a box in an observed frame, never a truth object ID."""
    agent_id: str
    image_timestamp_ns: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    event: str = "observe"

    def __post_init__(self):
        if self.agent_id not in AGENT_IDS or self.event not in REPORT_EVENTS:
            raise ValueError("Invalid report agent/event")
        integer(self.image_timestamp_ns, "image timestamp")
        object.__setattr__(self, "bbox_xyxy", bbox(self.bbox_xyxy))
        finite_number(self.confidence, "confidence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0,1]")

    def to_wire(self) -> dict:
        result = {"schema": GROUNDING_SCHEMA, **asdict(self)}
        result["bbox_xyxy"] = list(self.bbox_xyxy)
        assert_truth_free(result)
        return result

    @classmethod
    def from_wire(cls, payload: Mapping) -> "TargetReport":
        assert_truth_free(payload)
        exact_keys(payload, {"schema", "agent_id", "image_timestamp_ns", "bbox_xyxy", "confidence", "event"})
        if payload["schema"] != GROUNDING_SCHEMA:
            raise ValueError("Target report schema mismatch")
        return cls(payload["agent_id"], payload["image_timestamp_ns"], bbox(payload["bbox_xyxy"]),
                   payload["confidence"], payload["event"])
