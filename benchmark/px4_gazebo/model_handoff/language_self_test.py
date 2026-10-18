"""Synthetic no-flight contract tests, not a real visual grounding benchmark."""

from dataclasses import replace
import json
import unittest

from .fixtures import Appearance
from .grounded_interface import GroundedAction
from .language_evaluator import LanguageEvaluator, ObjectAnnotation, ObservedFrame
from .language_tasks import TargetReport, TaskRules, paired_target_tasks, templated_task
from .protocol import ModelJointAction, assert_truth_free

OBJECTS = {"private_red": Appearance("red", "sphere", "black"),
           "private_blue": Appearance("blue", "cube", "white")}
BOX = (.1, .1, .3, .3)
OTHER = (.6, .6, .8, .8)


def evaluator(task="handoff", descriptor=None):
    return LanguageEvaluator(templated_task(task, descriptor or OBJECTS["private_red"]).bind(OBJECTS))


def emit(ev, agent, t, event="observe", *, box=BOX, view=None, annotations=None, now=None):
    ns = round(t * 1e9)
    if annotations is None:
        annotations = {"private_red": ObjectAnnotation(BOX, view),
                       "private_blue": ObjectAnnotation(OTHER)}
    ev.register_observed_frame(ObservedFrame(agent, ns, ns, annotations))
    ev.report(TargetReport(agent, ns, box, .9, event), now_sim_time_ns=ns if now is None else round(now * 1e9))


def source_track(ev):
    for t in (0., .1, .2, .3, .4):
        emit(ev, "uav_0", t)
    emit(ev, "uav_0", .5, "handoff_request")


def complete_handoff(ev):
    source_track(ev)
    emit(ev, "uav_1", .6, "handoff_accept")
    for t in (.7, .8, .9, 1.):
        emit(ev, "uav_1", t)
    emit(ev, "uav_1", 1.1, "confirm")


class LanguageTests(unittest.TestCase):
    def test_binding_changes_answer(self):
        pair = paired_target_tasks("search", OBJECTS)
        self.assertNotEqual(pair[0].target_id, pair[1].target_id)
        self.assertNotEqual(pair[0].definition.instruction, pair[1].definition.instruction)
        for bound in pair:
            wire = bound.definition.public_context()
            assert_truth_free(wire)
            self.assertNotIn("descriptor", wire)
            self.assertNotIn(bound.target_id, json.dumps(wire))
        with self.assertRaises(ValueError):
            templated_task("search", OBJECTS["private_red"]).bind({"a": OBJECTS["private_red"], "b": OBJECTS["private_red"]})

    def test_strict_report_and_envelope(self):
        report = TargetReport("uav_0", 10, BOX, .8, "found")
        action = GroundedAction(ModelJointAction(), (report,))
        self.assertEqual(GroundedAction.from_wire(json.loads(json.dumps(action.to_wire()))), action)
        for key, value in (("target_id", "private_red"), ("unexpected", 1)):
            payload = report.to_wire()
            payload[key] = value
            with self.assertRaises(ValueError):
                TargetReport.from_wire(payload)
        for change in ({"confidence": True}, {"confidence": "0.8"}, {"image_timestamp_ns": True},
                       {"bbox_xyxy": (0., 0., 0., 1.)}, {"confidence": float("nan")}):
            with self.assertRaises(ValueError):
                replace(report, **change)
        for change in ({"min_iou": True}, {"max_report_age_s": "0.2"}, {"min_dwell_s": 0}):
            with self.assertRaises(ValueError):
                TaskRules(**change)
        with self.assertRaises(ValueError):
            GroundedAction(ModelJointAction(), (report, report))
        payload = action.to_wire()
        payload["action"]["unknown"] = 1
        with self.assertRaises(ValueError):
            GroundedAction.from_wire(payload)

    def test_search_and_changed_instruction(self):
        for descriptor, box, expected in ((None, BOX, "success"), (OBJECTS["private_blue"], BOX, "model_failure"),
                                         (OBJECTS["private_blue"], OTHER, "success")):
            ev = evaluator("search", descriptor)
            emit(ev, "uav_0", 0., "found", box=box)
            emit(ev, "uav_1", .1, "confirm", box=box)
            self.assertEqual(ev.finish()["category"], expected)
        ev = evaluator("search")
        emit(ev, "uav_1", 0., "confirm")
        self.assertEqual(ev.finish()["reason"], "invalid_search_event_order")

    def test_handoff_and_late_safety(self):
        ev = evaluator()
        complete_handoff(ev)
        self.assertEqual(ev.finish()["category"], "success")
        ev = evaluator()
        complete_handoff(ev)
        ev.safety_violation("collision")
        self.assertEqual(ev.finish()["category"], "model_failure")
        with self.assertRaises(RuntimeError):
            ev.safety_violation("late")

    def test_handoff_order_and_gap(self):
        ev = evaluator()
        emit(ev, "uav_1", 0., "handoff_accept")
        self.assertEqual(ev.finish()["reason"], "accept_without_request")
        ev = evaluator()
        emit(ev, "uav_0", 0.)
        emit(ev, "uav_0", .1)
        emit(ev, "uav_0", .6, "handoff_request")
        self.assertEqual(ev.finish()["reason"], "request_before_source_tracking")
        ev = evaluator()
        source_track(ev)
        emit(ev, "uav_1", .6, "handoff_accept")
        emit(ev, "uav_1", .7, "confirm")
        self.assertEqual(ev.finish()["reason"], "confirm_before_receiver_tracking")

    def test_repeat_frames_no_extra_events(self):
        ev = evaluator("search")
        emit(ev, "uav_0", 0., "found")
        emit(ev, "uav_1", .1)
        ev.report(TargetReport("uav_1", 100_000_000, BOX, .9, "confirm"), now_sim_time_ns=100_000_000)
        self.assertEqual(ev.finish()["reason"], "incomplete_task_evidence")
        ev = evaluator()
        emit(ev, "uav_0", 0.)
        for t in (.05, .1, .15, .2):
            ev.report(TargetReport("uav_0", 0, BOX, .9), now_sim_time_ns=round(t * 1e9))
        emit(ev, "uav_0", .5, "handoff_request")
        self.assertEqual(ev.finish()["reason"], "request_before_source_tracking")

    def test_invalid_grounding(self):
        cases = [(OTHER, None, None, "wrong_instruction_target"),
                 ((.35, .35, .45, .45), None, None, "ungrounded_target_report"),
                 (BOX, {key: ObjectAnnotation(BOX) for key in OBJECTS}, None, "ambiguous_target_report"),
                 (BOX, None, .3, "future_or_stale_frame"),
                 (BOX, None, -.1, "future_or_stale_frame")]
        for box, annotations, now, reason in cases:
            ev = evaluator("search")
            # Future frame check uses positive simulation timestamps.
            emit(ev, "uav_0", 1., "found", box=box, annotations=annotations,
                 now=None if now is None else 1. + now)
            self.assertEqual(ev.finish()["reason"], reason)
        ev = evaluator()
        ev.report(TargetReport("uav_0", 100, BOX, .9), now_sim_time_ns=100)
        self.assertEqual(ev.finish()["reason"], "unobserved_frame")

    def test_private_errors_are_not_model_errors(self):
        ev = evaluator("search")
        emit(ev, "uav_0", 0., "found", annotations={"private_red": ObjectAnnotation(BOX)})
        self.assertEqual(ev.finish()["category"], "evaluator_error")
        ev = evaluator("inspection")
        emit(ev, "uav_0", 0., "inspect")
        self.assertEqual(ev.finish()["reason"], "missing_inspection_geometry")
        ev = evaluator()
        ev.invalidate("simulator_error", "camera_timeout")
        self.assertEqual(ev.finish()["category"], "simulator_error")
        ev = evaluator()
        emit(ev, "uav_0", 0.)
        emit(ev, "uav_0", 0., annotations={key: ObjectAnnotation(OTHER) for key in OBJECTS})
        self.assertEqual(ev.finish()["reason"], "conflicting_frame_annotation")

    def test_inspection_complementary_views(self):
        for receiver_view, expected in (("side", "success"), ("front", "model_failure")):
            ev = evaluator("inspection")
            for t in (0., .1, .2, .3, .4, .5):
                emit(ev, "uav_0", t, "inspect", view="front")
                emit(ev, "uav_1", t, "inspect", view=receiver_view)
            emit(ev, "uav_0", .6, "confirm", view="front")
            emit(ev, "uav_1", .6, "confirm", view=receiver_view)
            self.assertEqual(ev.finish()["category"], expected)

    def test_interrupted_visibility(self):
        ev = evaluator()
        for t in (0., .1, .2, .3, .4):
            emit(ev, "uav_0", t)
        ev.register_observed_frame(ObservedFrame("uav_0", 450_000_000, 450_000_000,
            {"private_red": ObjectAnnotation(None), "private_blue": ObjectAnnotation(OTHER)}))
        emit(ev, "uav_0", .5, "handoff_request")
        self.assertEqual(ev.finish()["reason"], "request_before_source_tracking")

    def test_pre_event_image_cannot_confirm(self):
        ev = evaluator("search")
        emit(ev, "uav_0", 1., "found", now=1.1)
        emit(ev, "uav_1", 1.05, "confirm", now=1.1)
        self.assertEqual(ev.finish()["category"], "model_failure")

    def test_inspection_requires_both_contributions_and_confirmations(self):
        ev = evaluator("inspection")
        for start, view in ((0., "front"), (.6, "side")):
            for offset in (0., .1, .2, .3, .4, .5):
                emit(ev, "uav_0", start + offset, "inspect", view=view)
        emit(ev, "uav_1", 1.2, "confirm", view="side")
        self.assertEqual(ev.finish()["reason"], "confirm_before_dual_view_inspection")
        ev = evaluator("inspection")
        for t in (0., .1, .2, .3, .4, .5):
            emit(ev, "uav_0", t, "inspect", view="front")
            emit(ev, "uav_1", t, "inspect", view="side")
        emit(ev, "uav_0", .6, "confirm", view="front")
        self.assertEqual(ev.finish()["reason"], "incomplete_task_evidence")

    def test_old_receiver_images_do_not_inflate_post_accept_dwell(self):
        ev = evaluator()
        source_track(ev)
        emit(ev, "uav_1", .6, "handoff_accept", now=.7)
        emit(ev, "uav_1", .65, now=.7)
        self.assertEqual(ev.finish()["reason"], "pre_accept_tracking_frame")

    def test_annotation_snapshot_and_superseded_report(self):
        ev = evaluator("search")
        annotations = {"private_red": ObjectAnnotation(BOX), "private_blue": ObjectAnnotation(OTHER)}
        ev.register_observed_frame(ObservedFrame("uav_0", 0, 0, annotations))
        annotations["private_red"] = ObjectAnnotation(None)
        ev.report(TargetReport("uav_0", 0, BOX, .9, "found"), now_sim_time_ns=0)
        emit(ev, "uav_1", .1, "confirm")
        self.assertEqual(ev.finish()["category"], "success")
        ev = evaluator()
        for ns in (0, 100_000_000):
            ev.register_observed_frame(ObservedFrame("uav_0", ns, ns,
                {"private_red": ObjectAnnotation(BOX), "private_blue": ObjectAnnotation(OTHER)}))
        ev.report(TargetReport("uav_0", 0, BOX, .9), now_sim_time_ns=100_000_000)
        self.assertEqual(ev.finish()["reason"], "superseded_frame")


if __name__ == "__main__":
    unittest.main()
