"""Private, offline draft acceptance. Not a live annotation generator.

Frames must be registered only after delivery to the policy. sim_time_ns is
the privately aligned simulation capture time, not host UNIX/monotonic time.
No method in this module is a policy response or an action planner.
"""

from dataclasses import dataclass
from typing import Mapping

from .language_tasks import BoundTask, TargetReport, bbox, integer
from .protocol import AGENT_IDS

EVALUATOR_SCHEMA = "aerowam-language-evaluation/1-draft"


class AnnotationError(ValueError):
    """Evaluator evidence problem, never a model capability failure."""


@dataclass(frozen=True)
class ObjectAnnotation:
    # None means not visibly identifiable; never use an amodal box here.
    visible_bbox: tuple[float, float, float, float] | None
    inspection_view: str | None = None
    inspection_geometry_available: bool = False

    def __post_init__(self):
        if type(self.inspection_geometry_available) is not bool:
            raise AnnotationError("Invalid inspection geometry availability")
        if self.visible_bbox is not None:
            object.__setattr__(self, "visible_bbox", bbox(self.visible_bbox))
        if self.inspection_view is not None and (not isinstance(self.inspection_view, str) or not self.inspection_view):
            raise AnnotationError("Invalid private inspection view")
        if self.visible_bbox is None and self.inspection_view is not None:
            raise AnnotationError("Invisible object cannot provide inspection evidence")


@dataclass(frozen=True)
class ObservedFrame:
    agent_id: str
    image_timestamp_ns: int
    sim_time_ns: int
    objects: Mapping[str, ObjectAnnotation]

    def __post_init__(self):
        if self.agent_id not in AGENT_IDS:
            raise AnnotationError("Unknown camera")
        integer(self.image_timestamp_ns, "image stamp")
        integer(self.sim_time_ns, "simulation capture time")
        if not isinstance(self.objects, Mapping) or any(
                not isinstance(key, str) or not key or not isinstance(value, ObjectAnnotation)
                for key, value in self.objects.items()):
            raise AnnotationError("Invalid frame annotations")


def iou(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(a) + area(b) - intersection)


class LanguageEvaluator:
    """Private episode state. Success remains provisional until finish()."""

    def __init__(self, task: BoundTask):
        if not isinstance(task, BoundTask):
            raise AnnotationError("A privately bound task is required")
        self.task = task
        self.rules = task.definition.rules
        self.frames = {}
        self.latest = {}
        self.used = set()
        self.spans = {}
        self.found = self.requested = self.accepted = False
        self.confirmed = set()
        self.views = {agent: set() for agent in AGENT_IDS}
        self.completed = False
        self.failure = None
        self.safety_reason = None
        self.last_time = -1
        self.closed = False
        self.stage_time = -1

    def _fail(self, category, reason):
        # Missing/invalid evaluation evidence takes precedence over ability claims.
        if self.failure is None or category in {"evaluator_error", "simulator_error"}:
            self.failure = (category, reason)

    def register_observed_frame(self, frame: ObservedFrame):
        """Trusted harness call AFTER actual delivery; no policy-supplied truth."""
        if self.closed:
            raise RuntimeError("Episode already finalized")
        if not isinstance(frame, ObservedFrame):
            self._fail("evaluator_error", "invalid_frame")
            return
        if set(frame.objects) != set(self.task.object_ids):
            self._fail("evaluator_error", "missing_or_unknown_scene_identity")
            return
        if any(value.inspection_view is not None and value.inspection_view not in self.rules.inspection_views
               for value in frame.objects.values()):
            self._fail("evaluator_error", "unsupported_private_view")
            return
        # Snapshot mappings so later caller mutation cannot change past evidence.
        frame = ObservedFrame(frame.agent_id, frame.image_timestamp_ns, frame.sim_time_ns, dict(frame.objects))
        key = (frame.agent_id, frame.image_timestamp_ns)
        if key in self.frames:
            if self.frames[key] != frame:
                self._fail("evaluator_error", "conflicting_frame_annotation")
            return
        previous = self.latest.get(frame.agent_id)
        if previous is not None and (frame.image_timestamp_ns <= previous.image_timestamp_ns
                                     or frame.sim_time_ns <= previous.sim_time_ns):
            self._fail("evaluator_error", "nonadvancing_frame_clock")
            return
        self.frames[key] = frame
        self.latest[frame.agent_id] = frame
        if frame.objects[self.task.target_id].visible_bbox is None:
            self._reset_spans(frame.agent_id)
        elif self.task.definition.task == "inspection":
            annotation = frame.objects[self.task.target_id]
            if annotation.inspection_view is None:
                self._reset_spans(frame.agent_id)
            else:
                self.spans = {key: value for key, value in self.spans.items()
                              if key[0] != frame.agent_id or key[1] == annotation.inspection_view}

    def _reset_spans(self, agent):
        self.spans = {key: value for key, value in self.spans.items() if key[0] != agent}

    def _dwell(self, agent, view, t):
        key = (agent, view)
        start, end = self.spans.get(key, (t, t))
        if t - end > round(self.rules.max_evidence_gap_s * 1e9):
            start = t
        self.spans[key] = (start, t)
        return t - start >= round(self.rules.min_dwell_s * 1e9)

    def report(self, report: TargetReport, *, now_sim_time_ns: int):
        """Consume one report; never return matched truth identity to the policy."""
        if self.closed:
            raise RuntimeError("Episode already finalized")
        integer(now_sim_time_ns, "simulation report time")
        if now_sim_time_ns < self.last_time:
            self._fail("evaluator_error", "nonmonotonic_report_clock")
            return
        self.last_time = now_sim_time_ns
        if not isinstance(report, TargetReport):
            self._fail("model_failure", "invalid_target_report")
            return
        key = (report.agent_id, report.image_timestamp_ns)
        frame = self.frames.get(key)
        if frame is None:
            self._fail("model_failure", "unobserved_frame")
            return
        age = now_sim_time_ns - frame.sim_time_ns
        if age < 0 or age > round(self.rules.max_report_age_s * 1e9):
            self._fail("model_failure", "future_or_stale_frame")
            return
        if key in self.used:
            # A frame can support only one event, and no additional dwell.
            return
        if frame != self.latest[report.agent_id]:
            self._fail("model_failure", "superseded_frame")
            return
        self.used.add(key)
        scores = sorted(((iou(report.bbox_xyxy, value.visible_bbox), identity)
                         for identity, value in frame.objects.items() if value.visible_bbox is not None), reverse=True)
        if not scores or scores[0][0] < self.rules.min_iou:
            self._fail("model_failure", "ungrounded_target_report")
            self._reset_spans(report.agent_id)
            return
        if len(scores) > 1 and scores[0][0] - scores[1][0] < self.rules.ambiguity_margin:
            self._fail("model_failure", "ambiguous_target_report")
            return
        if scores[0][1] != self.task.target_id:
            self._fail("model_failure", "wrong_instruction_target")
            return
        annotation = frame.objects[self.task.target_id]
        if (self.task.definition.task == "inspection" and annotation.inspection_view is None
                and not annotation.inspection_geometry_available):
            self._fail("evaluator_error", "missing_inspection_geometry")
            return
        self._event(report, frame, annotation, now_sim_time_ns)

    def _event(self, report, frame, annotation, now):
        agent, event, t = report.agent_id, report.event, frame.sim_time_ns
        definition = self.task.definition
        source, receiver = definition.source_agent, definition.receiver_agent
        if definition.task == "search":
            if event == "found" and agent == source and not self.found:
                self.found = True
                self.stage_time = now
            elif event == "confirm" and agent == receiver and self.found and t >= self.stage_time:
                self.completed = True
            elif event != "observe":
                self._fail("model_failure", "invalid_search_event_order")
            return
        if definition.task == "handoff":
            if event not in {"observe", "handoff_request", "handoff_accept", "confirm"}:
                self._fail("model_failure", "invalid_handoff_event")
                return
            if agent == receiver and self.accepted and t < self.stage_time:
                self._fail("model_failure", "pre_accept_tracking_frame")
                return
            ready = self._dwell(agent, "track", t) if agent == source or self.accepted else False
            if event == "handoff_request":
                if agent != source or not ready or self.requested:
                    self._fail("model_failure", "request_before_source_tracking")
                else:
                    self.requested = True
                    self.stage_time = now
            elif event == "handoff_accept":
                if agent != receiver or not self.requested or self.accepted or t < self.stage_time:
                    self._fail("model_failure", "accept_without_request")
                else:
                    self.accepted = True
                    self.stage_time = now
                    self._reset_spans(receiver)
                    # Dwell starts no earlier than actual acceptance, even if
                    # the accept box refers to a slightly earlier image.
                    self._dwell(receiver, "track", now)
            elif event == "confirm":
                if agent != receiver or not self.accepted or not ready or t < self.stage_time:
                    self._fail("model_failure", "confirm_before_receiver_tracking")
                else:
                    self.completed = True
            return
        # Inspection views are independently computed by the private annotator.
        if event == "inspect":
            view = annotation.inspection_view
            if view is None:
                self._reset_spans(agent)
                return
            # Switching view breaks its previous continuous dwell.
            self.spans = {key: value for key, value in self.spans.items()
                          if key[0] != agent or key[1] == view}
            if self._dwell(agent, view, t):
                self.views[agent].add(view)
                self.stage_time = now
        elif event == "confirm":
            covered = set.union(*self.views.values()) >= set(self.rules.inspection_views)
            complementary = any(a != b for a in self.views[source] for b in self.views[receiver])
            if not covered or not complementary or t < self.stage_time:
                self._fail("model_failure", "confirm_before_dual_view_inspection")
            else:
                self.confirmed.add(agent)
                self.completed = self.confirmed == set(AGENT_IDS)
        elif event == "observe":
            self._reset_spans(agent)
        else:
            self._fail("model_failure", "invalid_inspection_event")

    def invalidate(self, category: str, reason: str):
        if self.closed:
            raise RuntimeError("Episode already finalized")
        if category not in {"simulator_error", "evaluator_error"} or not isinstance(reason, str) or not reason:
            raise ValueError("Explicit infrastructure/evaluator category required")
        self._fail(category, reason)

    def model_error(self, reason: str):
        if self.closed:
            raise RuntimeError("Episode already finalized")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Model failure reason required")
        self._fail("model_failure", reason)

    def safety_violation(self, reason: str):
        if self.closed:
            raise RuntimeError("Episode already finalized")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Private safety reason required")
        self.safety_reason = reason

    def finish(self) -> dict:
        self.closed = True
        if self.failure is not None:
            category, reason = self.failure
        elif self.safety_reason is not None:
            category, reason = "model_failure", self.safety_reason
        elif self.completed:
            category, reason = "success", "task_evidence_complete"
        else:
            category, reason = "model_failure", "incomplete_task_evidence"
        return {"schema": EVALUATOR_SCHEMA, "task": self.task.definition.task,
                "category": category, "reason": reason,
                "safety_violation": self.safety_reason,
                "task_evidence_complete": self.completed,
                "unique_report_frames": len(self.used)}
