"""No-flight command response, causality and public/private boundary tests."""

from copy import deepcopy
from dataclasses import replace
import unittest

from .grounded_interface import GroundedAction
from .protocol import ModelJointAction, ModelObservation, WaypointDelta
from .search_pilot_self_test import observation
from .teammate_response import ResponseSpec, TeammateResponse, audit_response_trace, request_control


def action(x=.4):
    return ModelJointAction(WaypointDelta(.1, -.2, .03, -.04),
                            WaypointDelta(x, -.2, .1, .08), "confirm")


class TeammateResponseTests(unittest.TestCase):
    def test_nominal_exact_noop(self):
        transform = TeammateResponse()
        request = action()
        issued, row = transform.issue(request, sim_stamp_ns=0)
        self.assertIs(issued, request)
        self.assertEqual(row["requested_action"], row["issued_action"])
        self.assertFalse(audit_response_trace(transform.spec, [row])["nontrivial_perturbation_observed"])

    def test_cautious_scales_only_peer_all_axes(self):
        transform = TeammateResponse(ResponseSpec("cautious"))
        request = action()
        issued, row = transform.issue(request, sim_stamp_ns=1)
        self.assertEqual(issued.uav_0, request.uav_0)
        self.assertEqual(issued.intent, request.intent)
        self.assertEqual(issued.uav_1.vector(), [.2, -.1, .05, .04])
        self.assertEqual(request, action())
        self.assertEqual(audit_response_trace(transform.spec, [row])["changed_peer_requests"], 1)

    def test_delayed_uses_simulation_time_and_latest_eligible(self):
        transform = TeammateResponse(ResponseSpec("delayed"))
        rows = []
        for stamp, x, expected in ((0, .1, 0.), (500_000_000, .2, 0.),
                                    (750_000_000, .3, .1), (1_500_000_000, .4, .3)):
            issued, row = transform.issue(action(x), sim_stamp_ns=stamp)
            self.assertEqual(issued.uav_1.dx, expected)
            self.assertEqual(issued.uav_0, action().uav_0)
            rows.append(row)
        self.assertEqual([r["peer_source_step"] for r in rows], [None, None, 1, 3])
        self.assertTrue(audit_response_trace(transform.spec, rows)["trace_valid"])

    def test_separate_episodes_have_no_pending_action_leak(self):
        first = TeammateResponse(ResponseSpec("delayed"))
        first.issue(action(), sim_stamp_ns=0)
        first.issue(action(), sim_stamp_ns=1_000_000_000)
        second = TeammateResponse(ResponseSpec("delayed"))
        issued, row = second.issue(action(), sim_stamp_ns=0)
        self.assertEqual(issued.uav_1, WaypointDelta())
        self.assertIsNone(row["peer_source_step"])

    def test_invalid_time_rejected_without_advancing(self):
        transform = TeammateResponse(ResponseSpec("delayed"))
        transform.issue(action(), sim_stamp_ns=100)
        for stamp in (100, 99, -1, True, 1.5):
            with self.assertRaises(ValueError):
                transform.issue(action(), sim_stamp_ns=stamp)
        self.assertEqual(transform.step, 1)
        self.assertEqual(len(transform.pending), 1)

    def test_pending_queue_bounded(self):
        transform = TeammateResponse(ResponseSpec("delayed"))
        for stamp in range(256):
            transform.issue(action(), sim_stamp_ns=stamp)
        with self.assertRaises(RuntimeError):
            transform.issue(action(), sim_stamp_ns=256)
        self.assertEqual(transform.step, 256)

    def test_bad_profile_and_type_rejected(self):
        with self.assertRaises(ValueError):
            ResponseSpec("oracle")
        with self.assertRaises(TypeError):
            TeammateResponse({"profile": "nominal"})
        with self.assertRaises(TypeError):
            TeammateResponse().issue({}, sim_stamp_ns=1)

    def test_empty_or_unchanged_trace_never_proves_intervention(self):
        spec = ResponseSpec("cautious")
        self.assertFalse(audit_response_trace(spec, [])["nontrivial_perturbation_observed"])
        _, row = TeammateResponse(spec).issue(ModelJointAction(), sim_stamp_ns=1)
        result = audit_response_trace(spec, [row])
        self.assertFalse(result["nontrivial_perturbation_observed"])
        self.assertFalse(result["physical_execution_verified"])
        self.assertFalse(result["task_success_evaluated"])

    def test_audit_rejects_forged_profile_causality_and_commands(self):
        spec = ResponseSpec("delayed")
        transform = TeammateResponse(spec)
        rows = [transform.issue(action(x), sim_stamp_ns=t)[1]
                for t, x in ((0, .1), (500_000_000, .2), (1_000_000_000, .3))]
        for mutate in (
            lambda r: r[0].update(peer_source_step=1),
            lambda r: r[2].update(peer_source_step=2),
            lambda r: r[1].update(sim_stamp_ns=0),
            lambda r: r[0].update(step=True),
            lambda r: r[0]["issued_action"]["waypoints"]["uav_0"].__setitem__(0, .2),
            lambda r: r[0]["issued_action"].update(intent="none"),
            lambda r: r[0]["issued_action"]["waypoints"]["uav_1"].__setitem__(0, .1),
            lambda r: r[0].update(target_id="hidden"),
        ):
            corrupted = deepcopy(rows)
            mutate(corrupted)
            with self.assertRaises(ValueError):
                audit_response_trace(spec, corrupted)
        with self.assertRaises(ValueError):
            audit_response_trace(ResponseSpec(), rows)

    def test_public_action_history_contains_issued_not_private_profile(self):
        obs = observation()
        issued, row = TeammateResponse(ResponseSpec("cautious")).issue(action(), sim_stamp_ns=1)
        obs = replace(obs, action_history=[row["issued_action"]["waypoints"]])
        decoded = ModelObservation.from_wire(obs.to_wire())
        self.assertEqual(decoded.action_history[-1]["uav_1"], issued.uav_1.vector())
        # Private profile, schedule and source index are not policy inputs.
        public = obs.canonical_bytes().decode()
        self.assertNotIn("cautious", public)
        self.assertNotIn("peer_source_step", public)
        self.assertNotIn("target_id", public)
        self.assertEqual(GroundedAction(action()).reports, ())

    def test_controller_boundary_and_reports_stay_separate(self):
        from .search_reporter import PilotReporter
        obs = observation()
        response = GroundedAction(action(), PilotReporter().reports(obs))
        class Control:
            def request(self, issued):
                self.received = issued
        control, trace = Control(), []
        issued = request_control(control, TeammateResponse(ResponseSpec("delayed")),
                                 response.action, sim_stamp_ns=obs.timestamp_ns, trace=trace)
        self.assertIs(control.received, issued)
        self.assertEqual(control.received.uav_1, WaypointDelta())
        self.assertEqual(response.action, action())
        self.assertTrue(response.reports)
        self.assertEqual(response.reports[0].image_timestamp_ns, obs.timestamp_ns)
        self.assertEqual(len(trace), 1)

    def test_rejected_controller_request_is_not_logged_as_issued(self):
        class RejectingControl:
            def request(self, issued):
                raise RuntimeError("controller rejected")
        trace = []
        with self.assertRaises(RuntimeError):
            request_control(RejectingControl(), TeammateResponse(), action(), sim_stamp_ns=1, trace=trace)
        self.assertEqual(trace, [])

    def test_audit_crosschecks_recorder_and_detects_dropped_final_row(self):
        transform = TeammateResponse(ResponseSpec("cautious"))
        rows = [transform.issue(action(x), sim_stamp_ns=t)[1] for t, x in ((1, .2), (2, .4))]
        records = [(r["requested_action"], r["issued_action"]) for r in rows]
        result = audit_response_trace(transform.spec, rows, controller_records=records)
        self.assertTrue(result["controller_records_cross_checked"])
        with self.assertRaises(ValueError):
            audit_response_trace(transform.spec, rows[:-1], controller_records=records)
        with self.assertRaises(ValueError):
            audit_response_trace(transform.spec, rows, controller_records=records[::-1])

    def test_private_behavior_or_truth_in_messages_never_reaches_adapter(self):
        from .grounded_interface import GroundedAdapter
        from .decision_interface import DecisionAdapter
        from .interface_self_test import _decision_request
        calls = []
        grounded = GroundedAdapter(lambda obs: calls.append(obs))
        class Predictor:
            def predict(self, **kwargs):
                calls.append(kwargs)
        decision = DecisionAdapter(Predictor())
        keys = ("teammate_response", "teammate_policy_id", "teammate_behavior_id",
                "teammate_response_schedule", "motion_calibration", "private_local_goals",
                "candidate_actual_utility", "target_visibility_truth", "target_pose", "future_trajectory")
        for key in keys:
            for adapter, payload in ((grounded, observation().to_wire()), (decision, _decision_request().to_wire())):
                obs = payload if adapter is grounded else payload["history"][0]
                obs["message_history"] = [{"payload": {key: "hidden"}}]
                with self.assertRaises(ValueError):
                    adapter(payload)
        self.assertFalse(calls)

    def test_behavior_variation_cannot_substitute_for_visible_target_evidence(self):
        from .language_evaluator import ObjectAnnotation, ObservedFrame
        from .language_tasks import TargetReport
        from .search_episode import SearchEpisode
        from .search_pilot_scene import manual_condition
        obs = observation()
        episode = SearchEpisode(manual_condition(obs.instruction, "red"), obs.episode_id)
        frames = {a: ObservedFrame(a, obs.timestamp_ns, obs.timestamp_ns, {
            "probe_red": ObjectAnnotation(None), "probe_blue": ObjectAnnotation(None)}) for a in obs.states}
        reports = (TargetReport("uav_0", obs.timestamp_ns, (.1, .1, .3, .3), 1., "found"),)
        issued, _ = TeammateResponse(ResponseSpec("cautious")).issue(action(), sim_stamp_ns=1)
        episode.delivered(obs, frames, GroundedAction(issued, reports), now_sim_time_ns=obs.timestamp_ns)
        self.assertNotEqual(episode.finish()["category"], "success")


if __name__ == "__main__":
    unittest.main()
