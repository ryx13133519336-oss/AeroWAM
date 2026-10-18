"""Synthetic role-free movement/report/runner checks, not model ability or flight."""

from dataclasses import replace
import threading
import unittest
from unittest.mock import patch

from .grounded_interface import GroundedAction
from .language_evaluator import ObjectAnnotation, ObservedFrame
from .language_tasks import TargetReport
from .policy_client import PolicyClient, PolicyRemoteError, PolicyTimeoutError
from .policy_server import ReusableThreadingTCPServer, _Handler, search_composite
from .protocol import AGENT_IDS, ModelJointAction, VehicleState, WaypointDelta
from .role_free_search import (RoleFreeApproachReporter, RoleFreeMotionSearchGate,
                               ROLE_FREE_REPORTER_VERSION, ROLE_FREE_MOTION_VERSION)
from .search_episode import SearchEpisode
from .search_motion_gate import MotionSearchGate
from .search_pilot import validate_search_role_mode, search_evidence_versions, policy_failure_record
from .search_reporter import SearchTravelBudgetExceeded
from .search_pilot_scene import manual_condition, paired_conditions
from .search_pilot_self_test import observation

MODULE = "benchmark.px4_gazebo.model_handoff.role_free_search.color_box"


def box(width):
    return (.5 - width / 2, .4, .5 + width / 2, .6)


def capture(t, source="uav_1", source_x=0., peer_x=0., speed=0.):
    stamp = round(t * 1e9)
    return replace(observation(stamp=stamp), states={
        a: VehicleState((source_x if a == source else peer_x, 0., 2.), (speed, 0., 0.), 0., "ok")
        for a in AGENT_IDS})


def emit(reporter, obs, widths=(.12, .12)):
    with patch(MODULE, side_effect=[None if w is None else box(w) for w in widths]):
        return reporter.reports(obs)


def frames(obs, widths=(.12, .12), target="probe_red"):
    return {a: ObservedFrame(a, obs.image_timestamp_ns[a], obs.image_timestamp_ns[a], {
        key: ObjectAnnotation(box(w) if key == target and w is not None else None)
        for key in ("probe_red", "probe_blue")}) for a, w in zip(AGENT_IDS, widths)}


class RoleFreeSearchTests(unittest.TestCase):
    def test_exact_travel_boundary_and_both_offenders_retained_without_evidence(self):
        for source in AGENT_IDS:
            reporter = RoleFreeApproachReporter()
            emit(reporter, capture(1., source), (None, None))
            emit(reporter, capture(2., source, 2.5, 2.5), (None, None))
            with self.assertRaises(SearchTravelBudgetExceeded) as caught:
                emit(reporter, capture(3., source, 2.501, -2.6), (None, None))
            self.assertEqual(caught.exception.diagnostics["exceeded_agents"], list(AGENT_IDS))
            self.assertEqual(caught.exception.diagnostics["limit_m"], 2.5)
            self.assertIsNone(reporter.source)
            self.assertFalse(reporter.confirmed)
            self.assertEqual(reporter.previous, capture(2.).image_timestamp_ns)

    def test_tcp_travel_rejection_preserves_terminal_input_and_no_applied_action(self):
        adapter, _ = search_composite(lambda request: ModelJointAction().to_wire(),
                                      approach=True, role_mode="unassigned")
        with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
            server.policy = adapter
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PolicyClient(f"127.0.0.1:{server.server_address[1]}", timeout_s=2.)
                with patch(MODULE, return_value=None):
                    client.infer_grounded(capture(1.))
                    terminal = capture(2., source="uav_0", peer_x=2.6)
                    with self.assertRaises(PolicyRemoteError) as caught:
                        client.infer_grounded(terminal)
                reason, record = policy_failure_record(terminal, caught.exception, 25.)
                self.assertEqual(reason, "search_local_travel_budget_exceeded")
                self.assertEqual(record["observation"], terminal.to_wire())
                self.assertEqual(record["diagnostics"]["exceeded_agents"], ["uav_1"])
                self.assertIsNone(record["response"])
                self.assertFalse(record["action_applied"])
                self.assertNotIn("inference_latency_ms", record)
            finally:
                server.shutdown()
                thread.join(timeout=2.)

    def test_untyped_error_or_timeout_never_becomes_travel_exhaustion(self):
        for error in (RuntimeError("RGB search local travel budget exceeded"),
                      PolicyTimeoutError("deadline expired"),
                      PolicyRemoteError({"error": "OtherError", "failure_code": SearchTravelBudgetExceeded.code})):
            reason, record = policy_failure_record(capture(1.), error, 401.)
            self.assertEqual(reason, "policy_inference_error")
            self.assertFalse(record["action_applied"])

    def test_task_binding_keeps_instruction_and_no_roles(self):
        for color in ("red", "blue"):
            task = manual_condition("原文不被改写", color, role_mode="unassigned")
            self.assertEqual(task.definition.instruction, "原文不被改写")
            self.assertEqual(task.target_id, "probe_" + color)
            self.assertNotIn("roles", task.definition.public_context())
        for task in paired_conditions(role_mode="unassigned"):
            self.assertEqual(task.definition.role_mode, "unassigned")
        with self.assertRaises(ValueError):
            manual_condition("red", "red", role_mode="bad")

    def test_runner_rejects_unmigrated_fixed_role_paths(self):
        validate_search_role_mode("unassigned", "checkpoint_search")
        self.assertEqual(search_evidence_versions("checkpoint_search", "unassigned"),
                         (ROLE_FREE_REPORTER_VERSION, ROLE_FREE_MOTION_VERSION))
        for policy in ("rgb_scan", "rgb_search", "checkpoint"):
            with self.assertRaises(ValueError):
                validate_search_role_mode("unassigned", policy)
        for profile in ("cautious", "delayed"):
            with self.assertRaises(ValueError):
                validate_search_role_mode("unassigned", "checkpoint_search", profile)
        with self.assertRaises(ValueError):
            search_composite(lambda r: ModelJointAction().to_wire(), role_mode="unassigned")

    def test_motion_gate_role_guard_also_applies_to_late_assignment(self):
        task = manual_condition("red", "red", role_mode="unassigned")
        ev = SearchEpisode(task, "opaque")
        with self.assertRaises(ValueError):
            ev.motion_gate = MotionSearchGate({})
        ev.motion_gate = RoleFreeMotionSearchGate({})
        with self.assertRaises(ValueError):
            SearchEpisode(manual_condition("red", "red"), "opaque", ev.motion_gate)

    def test_both_discovery_directions_and_colors_complete_same_thresholds(self):
        for source in AGENT_IDS:
            for color in ("red", "blue"):
                reporter = RoleFreeApproachReporter()
                task = manual_condition("find " + color, color, role_mode="unassigned")
                ev = SearchEpisode(task, "opaque", RoleFreeMotionSearchGate({a: (4., 0., 2.) for a in AGENT_IDS}))
                def step(t, sx=0., px=0., sw=.1, pw=.1):
                    obs = replace(capture(t, source, sx, px), instruction=task.definition.instruction)
                    widths = tuple(sw if a == source else pw for a in AGENT_IDS)
                    reports = emit(reporter, obs, widths)
                    ev.delivered(obs, frames(obs, widths, task.target_id), GroundedAction(ModelJointAction(), reports),
                                 now_sim_time_ns=obs.timestamp_ns)
                    self.assertIsNone(ev.evaluator.failure)
                    return reports
                step(1.)
                for t in (1.5, 2., 2.5, 3.):
                    reports = step(t, 1.12, 1.2, .12, .1)
                self.assertEqual([(r.agent_id, r.event) for r in reports if r.event != "observe"], [(source, "found")])
                self.assertEqual(reporter.source, source)
                for t in (3.5, 4., 4.5, 5.):
                    self.assertFalse(any(r.event == "confirm" for r in step(t, 1.12, 1.2, .12, .1)))
                for t in (5.5, 6., 6.5, 7.):
                    reports = step(t, 1.12, 2.32, .12, .12)
                self.assertIn("confirm", [r.event for r in reports])
                result = ev.finish()
                self.assertEqual(result["category"], "success")
                self.assertEqual(result["motion_acceptance"], ROLE_FREE_MOTION_VERSION)
                self.assertEqual(result["evidence_roles"]["source"], source)
                self.assertAlmostEqual(result["motion_metrics"]["receiver_displacement_after_found_m"], 1.12)

    def test_simultaneous_eligibility_has_no_id_tiebreak(self):
        reporter = RoleFreeApproachReporter()
        emit(reporter, capture(1.), (.1, .1))
        for t in (1.5, 2., 2.5, 3., 3.5):
            reports = emit(reporter, capture(t, source_x=1.12, peer_x=1.12))
            self.assertTrue(all(r.event == "observe" for r in reports))
        self.assertIsNone(reporter.source)
        self.assertTrue(all(d["phase"] == "SIMULTANEOUS_EVIDENCE_DEFERRED" for d in reporter.diagnostics.values()))
        reports = emit(reporter, capture(4., source_x=1.12, peer_x=1.12), (None, .12))
        self.assertEqual([(r.agent_id, r.event) for r in reports], [("uav_1", "found")])

    def test_stationary_fast_or_no_growth_cannot_discover_either_agent(self):
        for fault in ("stationary", "fast", "no_growth"):
            reporter = RoleFreeApproachReporter()
            emit(reporter, capture(1.), (.1, .1))
            for t in (1.5, 2., 2.5, 3., 3.5):
                reports = emit(reporter, capture(t, source_x=0. if fault == "stationary" else 1.12,
                                                speed=.3 if fault == "fast" else 0.),
                               (.1, .1) if fault == "no_growth" else (.12, .12))
                self.assertFalse(any(r.event != "observe" for r in reports))

    def test_stale_text_travel_and_reset(self):
        reporter = RoleFreeApproachReporter()
        emit(reporter, capture(1.))
        for obs in (capture(1.), replace(capture(1.5), instruction="blue")):
            with self.assertRaises(ValueError):
                emit(reporter, obs)
        with self.assertRaises(RuntimeError):
            emit(reporter, capture(2., source_x=2.6))
        emit(reporter, replace(capture(1.), episode_id="new"))
        self.assertIsNone(reporter.source)
        self.assertFalse(reporter.confirmed)

    def test_occlusion_or_gap_resets_reporter_dwell(self):
        for fault in ("occluded", "gap"):
            reporter = RoleFreeApproachReporter()
            emit(reporter, capture(1.), (.1, .1))
            emit(reporter, capture(1.5, source_x=1.12))
            emit(reporter, capture(2., source_x=1.12))
            if fault == "occluded":
                emit(reporter, capture(2.5, source_x=1.12), (.12, None))
            reports = emit(reporter, capture(3., source_x=1.12))
            self.assertFalse(any(r.event == "found" for r in reports))
            self.assertEqual(reporter.diagnostics["uav_1"]["stable_frames"], 1)

    def test_private_gate_rejects_motion_away_even_if_reporter_fooled(self):
        for source in AGENT_IDS:
            gate = RoleFreeMotionSearchGate({a: (-4., 0., 2.) for a in AGENT_IDS})
            initial = capture(1., source)
            gate.check(initial, frames(initial), (), "probe_red", initial.timestamp_ns)
            obs = capture(2., source, 1.12)
            report = TargetReport(source, obs.timestamp_ns, box(.12), 1., "found")
            self.assertEqual(gate.check(obs, frames(obs), (report,), "probe_red", obs.timestamp_ns),
                             "found_without_actual_source_approach")

    def test_self_confirm_and_double_found_rejected(self):
        for source in AGENT_IDS:
            gate = RoleFreeMotionSearchGate({a: (4., 0., 2.) for a in AGENT_IDS})
            obs = capture(1., source)
            gate.check(obs, frames(obs), (), "probe_red", obs.timestamp_ns)
            obs = capture(2., source, 1.12)
            report = lambda a, event: TargetReport(a, obs.timestamp_ns, box(.12), 1., event)
            self.assertEqual(gate.check(obs, frames(obs), tuple(report(a, "found") for a in AGENT_IDS),
                                        "probe_red", obs.timestamp_ns), "multiple_found_in_one_response")
            self.assertIsNone(gate.source)
            gate.check(obs, frames(obs), (report(source, "found"),), "probe_red", obs.timestamp_ns)
            self.assertEqual(gate.check(obs, frames(obs), (report(source, "confirm"),),
                                        "probe_red", obs.timestamp_ns), "invalid_search_event_order")

    def test_tcp_keeps_asymmetric_model_actions_and_public_request(self):
        calls = []
        action = ModelJointAction(uav_0=WaypointDelta(-.2, .1, .03, -.02),
                                  uav_1=WaypointDelta(.15, -.1, -.03, .04), intent="handoff")
        def model(request):
            calls.append(request)
            self.assertNotIn("target_id", request)
            self.assertNotIn("roles", request)
            return action.to_wire()
        adapter, version = search_composite(model, approach=True, role_mode="unassigned")
        self.assertEqual(version, ROLE_FREE_REPORTER_VERSION)
        with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
            server.policy = adapter
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PolicyClient(f"127.0.0.1:{server.server_address[1]}", timeout_s=2.)
                with patch(MODULE, side_effect=[box(.1), box(.1)]):
                    response, _ = client.infer_grounded(capture(1.))
            finally:
                server.shutdown()
                thread.join(timeout=2.)
        self.assertEqual(response.action, action)
        self.assertEqual(len(response.reports), 2)
        self.assertEqual(calls[0]["instruction"], capture(1.).instruction)

    def test_private_confirm_needs_post_found_motion_and_new_stable_frames(self):
        for source in AGENT_IDS:
            peer = next(a for a in AGENT_IDS if a != source)
            for fault in ("pre_found_only", "away", "fast", "no_dwell", "old_frame"):
                gate = RoleFreeMotionSearchGate({a: (4., 0., 2.) for a in AGENT_IDS})
                obs = capture(1., source)
                gate.check(obs, frames(obs), (), "probe_red", obs.timestamp_ns)
                obs = capture(2., source, 1.12, 1.2)
                found = TargetReport(source, obs.timestamp_ns, box(.12), 1., "found")
                self.assertIsNone(gate.check(obs, frames(obs), (found,), "probe_red", obs.timestamp_ns))
                peer_x = 1.2 if fault == "pre_found_only" else 0. if fault == "away" else 2.32
                obs = capture(3., source, 1.12, peer_x, speed=.3 if fault == "fast" else 0.)
                if fault == "old_frame":
                    obs = replace(obs, timestamp_ns=2_000_000_000,
                                  image_timestamp_ns={a: 2_000_000_000 for a in AGENT_IDS})
                confirm = TargetReport(peer, obs.image_timestamp_ns[peer], box(.12), 1., "confirm")
                self.assertEqual(gate.check(obs, frames(obs), (confirm,), "probe_red", obs.timestamp_ns),
                                 "confirm_without_actual_receiver_approach_and_dwell", fault)

    def test_private_gap_or_missing_report_or_visibility_breaks_dwell(self):
        for source in AGENT_IDS:
            peer = next(a for a in AGENT_IDS if a != source)
            for fault in ("gap", "missing_report", "invisible", "offcenter"):
                gate = RoleFreeMotionSearchGate({a: (4., 0., 2.) for a in AGENT_IDS})
                obs = capture(1., source)
                gate.check(obs, frames(obs), (), "probe_red", obs.timestamp_ns)
                obs = capture(2., source, 1.12)
                gate.check(obs, frames(obs), (TargetReport(source, obs.timestamp_ns, box(.12), 1., "found"),),
                           "probe_red", obs.timestamp_ns)
                def check(t, event="observe", absent=False, private_box=box(.12)):
                    obs = capture(t, source, 1.12, 1.12)
                    annotation = frames(obs)
                    annotation[peer] = replace(annotation[peer], objects={
                        "probe_red": ObjectAnnotation(private_box), "probe_blue": ObjectAnnotation(None)})
                    reports = () if absent else (TargetReport(peer, obs.timestamp_ns, box(.12), 1., event),)
                    return gate.check(obs, annotation, reports, "probe_red", obs.timestamp_ns)
                check(2.5)
                check(3.)
                if fault == "missing_report":
                    check(3.5, absent=True)
                elif fault == "invisible":
                    check(3.5, private_box=None)
                elif fault == "offcenter":
                    check(3.5, private_box=(.7, .4, .8, .6))
                self.assertEqual(check(4., "confirm"), "confirm_without_actual_receiver_approach_and_dwell")

    def test_wrong_object_is_still_rejected_by_independent_language_evaluator(self):
        task = manual_condition("find red", "red", role_mode="unassigned")
        ev = SearchEpisode(task, "opaque", RoleFreeMotionSearchGate({a: (4., 0., 2.) for a in AGENT_IDS}))
        obs = replace(capture(1.), instruction=task.definition.instruction)
        ev.delivered(obs, frames(obs), GroundedAction(ModelJointAction()), now_sim_time_ns=obs.timestamp_ns)
        obs = replace(capture(2., source_x=1.12), instruction=task.definition.instruction)
        wrong = {a: replace(f, objects={"probe_red": ObjectAnnotation((.1, .1, .2, .2)),
                                      "probe_blue": ObjectAnnotation(box(.12))})
                 for a, f in frames(obs).items()}
        ev.delivered(obs, wrong, GroundedAction(ModelJointAction(), (
            TargetReport("uav_1", obs.timestamp_ns, box(.12), 1., "found"),)), now_sim_time_ns=obs.timestamp_ns)
        result = ev.finish()
        self.assertEqual(result["reason"], "wrong_instruction_target")
        self.assertIsNone(result["evidence_roles"]["source"])


if __name__ == "__main__":
    unittest.main()
