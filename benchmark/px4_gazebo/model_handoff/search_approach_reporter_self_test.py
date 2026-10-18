"""Public evidence reporting, independent private acceptance, no flight."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from .grounded_interface import GroundedAction
from .language_evaluator import ObjectAnnotation, ObservedFrame
from .policy_server import search_composite
from .protocol import AGENT_IDS, ModelJointAction, VehicleState, WaypointDelta
from .search_approach_reporter import ApproachReporter, APPROACH_REPORTER_VERSION
from .search_episode import SearchEpisode
from .search_motion_gate import MotionSearchGate
from .search_pilot import CHECKPOINT_POLICIES, MOTION_POLICIES, resolve_run_duration
from .search_pilot_scene import manual_condition
from .search_pilot_self_test import observation

MODULE = "benchmark.px4_gazebo.model_handoff.search_approach_reporter.color_box"


def obs(t, scout=0., receiver=0., speed=0., episode="opaque"):
    return replace(observation(stamp=round(t * 1e9), episode=episode), states={
        a: VehicleState((x, 0., 2.), (speed, 0., 0.), 0., "ok")
        for a, x in zip(AGENT_IDS, (scout, receiver))})


def boxes(width):
    return (.5 - width / 2, .4, .5 + width / 2, .6)


def emit(reporter, observation, width=.12):
    with patch(MODULE, return_value=boxes(width)):
        return reporter.reports(observation)


class ApproachReporterTests(unittest.TestCase):
    def test_stationary_or_fast_or_no_image_growth_never_found(self):
        for fault in ("stationary", "fast", "no_growth"):
            reporter = ApproachReporter()
            emit(reporter, obs(1.), .1)
            for k in range(8):
                reports = emit(reporter, obs(1.5 + .5 * k,
                                            scout=0. if fault == "stationary" else 1.12,
                                            speed=.3 if fault == "fast" else 0.),
                               .1 if fault == "no_growth" else .12)
                self.assertFalse(any(r.event != "observe" for r in reports), fault)

    def test_edge_or_offcenter_candidate_has_no_report(self):
        for box in ((226/256, 91/256, 252/256, 137/256), (.7, .4, .8, .6), None):
            reporter = ApproachReporter()
            with patch(MODULE, return_value=box):
                self.assertFalse(reporter.reports(obs(1.)))

    def test_missing_or_gap_resets_stable_dwell(self):
        for fault in ("missing", "gap"):
            reporter = ApproachReporter()
            emit(reporter, obs(1.), .1)
            emit(reporter, obs(1.5, scout=1.12))
            emit(reporter, obs(2., scout=1.12))
            if fault == "missing":
                with patch(MODULE, return_value=None):
                    reporter.reports(obs(2.5, scout=1.12))
            reports = emit(reporter, obs(3., scout=1.12))
            self.assertFalse(any(r.event == "found" for r in reports))
            self.assertEqual(reporter.diagnostics["uav_0"]["stable_frames"], 1)

    def test_stale_frames_changed_text_and_episode_reset(self):
        reporter = ApproachReporter()
        emit(reporter, obs(1.), .1)
        with self.assertRaises(ValueError):
            emit(reporter, obs(1.))
        with self.assertRaises(ValueError):
            emit(reporter, replace(obs(1.5), instruction="找蓝色方块"))
        with self.assertRaises(RuntimeError):
            emit(reporter, obs(2., scout=2.6))
        emit(reporter, obs(1., episode="fresh"), .1)
        self.assertIsNone(reporter.found_stamp)
        self.assertFalse(reporter.confirmed)

    def test_found_then_receiver_movement_and_private_gate_complete(self):
        reporter = ApproachReporter()
        episode = SearchEpisode(manual_condition(obs(1.).instruction, "red"), "opaque",
                                MotionSearchGate({a: (4., 0., 2.4) for a in AGENT_IDS}))
        def step(o, width):
            reports = emit(reporter, o, width)
            frames = {a: ObservedFrame(a, o.timestamp_ns, o.timestamp_ns, {
                "probe_red": ObjectAnnotation(boxes(width)), "probe_blue": ObjectAnnotation(None)}) for a in AGENT_IDS}
            episode.delivered(o, frames, GroundedAction(ModelJointAction(), reports), now_sim_time_ns=o.timestamp_ns)
            self.assertIsNone(episode.evaluator.failure)
            return reports
        step(obs(1.), .1)
        for t in (1.5, 2., 2.5, 3.):
            reports = step(obs(t, scout=1.12, receiver=1.2), .12)
        self.assertEqual([r.event for r in reports], ["found"])
        self.assertFalse(reporter.confirmed)
        # Significant movement before source found is not receiver credit.
        for t in (3.5, 4., 4.5, 5.):
            reports = step(obs(t, scout=1.12, receiver=1.2), .12)
            self.assertFalse(any(r.event == "confirm" for r in reports))
        for t in (5.5, 6., 6.5, 7.):
            reports = step(obs(t, scout=1.12, receiver=2.32), .144)
        self.assertIn("confirm", [r.event for r in reports])
        result = episode.finish()
        self.assertEqual(result["category"], "success")
        self.assertAlmostEqual(result["motion_metrics"]["receiver_displacement_after_found_m"], 1.12)

    def test_actual_tcp_wrapper_keeps_all_model_actions_and_raw_text(self):
        import threading
        from .policy_client import PolicyClient
        from .policy_server import ReusableThreadingTCPServer, _Handler
        calls = []
        action = ModelJointAction(uav_0=WaypointDelta(-.2, .1, .03, -.02), intent="handoff")
        def model(request):
            calls.append(request)
            self.assertNotIn("truth", request)
            self.assertNotIn("target_id", request)
            return action.to_wire()
        adapter, version = search_composite(model, approach=True)
        self.assertEqual(version, APPROACH_REPORTER_VERSION)
        with ReusableThreadingTCPServer(("127.0.0.1", 0), _Handler) as server:
            server.policy = adapter
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PolicyClient(f"127.0.0.1:{server.server_address[1]}", timeout_s=2.)
                with patch(MODULE, return_value=boxes(.1)):
                    response, _ = client.infer_grounded(obs(1.))
            finally:
                server.shutdown()
                thread.join(timeout=2.)
        self.assertEqual(response.action, action)
        self.assertEqual([r.event for r in response.reports], ["observe"])
        self.assertEqual(calls[0]["instruction"], obs(1.).instruction)
        self.assertEqual(calls[0]["states"], json.loads(json.dumps(obs(1.).to_wire()["states"])))

    def test_new_mode_has_same_motion_gate_and_bounded_budget_family(self):
        self.assertIn("checkpoint_search", CHECKPOINT_POLICIES)
        self.assertIn("checkpoint_search", MOTION_POLICIES)
        self.assertNotIn("checkpoint", MOTION_POLICIES)
        self.assertEqual(resolve_run_duration("checkpoint_search"), 90.)
        with self.assertRaises(ValueError):
            resolve_run_duration("checkpoint_search", 121.)


if __name__ == "__main__":
    unittest.main()
