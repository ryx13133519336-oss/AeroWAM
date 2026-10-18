"""Moving-search regression and kinematic checks; not SITL flight evidence."""

from dataclasses import replace
from io import BytesIO
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from .grounded_interface import GroundedAction
from .language_evaluator import ObjectAnnotation, ObservedFrame
from .language_tasks import TargetReport
from .protocol import AGENT_IDS, ModelJointAction, VehicleState
from .search_episode import SearchEpisode
from .search_flight_control import SearchFlightControl
from .search_motion_gate import MotionSearchGate, private_local_goals
from .search_motion_policy import RGBSearchPolicy, hold, reportable_box
from .search_pilot_scene import manual_condition
from .search_pilot_self_test import observation, private_frames, RED_BOX
from .search_reporter import RGBScanPolicy, color_box


def at(stamp, scout=0., receiver=0., episode="opaque", speed=0.):
    obs = observation(stamp=stamp, episode=episode)
    return replace(obs, states={a: VehicleState((x, 0., 2.), (speed, 0., 0.), 0., "ok")
                                for a, x in zip(AGENT_IDS, (scout, receiver))})


def report(obs, agent, event):
    return TargetReport(agent, obs.timestamp_ns, RED_BOX, 1., event)


def gate_episode():
    obs = at(1_000_000_000)
    # Private fixtures: separate local frames, not a shared position input.
    gate = MotionSearchGate({a: (4., 0., 2.4) for a in AGENT_IDS})
    episode = SearchEpisode(manual_condition(obs.instruction, "red"), obs.episode_id, gate)
    deliver(episode, obs)
    return episode


def deliver(episode, obs, *reports):
    episode.delivered(obs, private_frames(obs), GroundedAction(ModelJointAction(), reports),
                      now_sim_time_ns=obs.timestamp_ns)


class MotionSearchTests(unittest.TestCase):
    def test_white_striped_edge_candidates_turn_without_reporting(self):
        # The non-blue stripe reaches the raster edge, while the blue support
        # stops inside it. Exact color-mask border rejection misses this case.
        for shape, blue, yaw_sign in (
                ((226, 91, 260, 137), (226, 91, 250, 137), -1),
                ((-5, 91, 30, 137), (5, 91, 30, 137), 1),
                ((112, -5, 142, 35), (112, 5, 142, 35), 0),
                ((112, 220, 142, 260), (112, 220, 142, 250), 0)):
            with self.subTest(shape=shape):
                image = Image.new("RGB", (256, 256), (80, 80, 80))
                draw = ImageDraw.Draw(image)
                draw.rectangle(shape, fill=(245, 245, 245))
                draw.rectangle(blue, fill=(4, 4, 180))
                stream = BytesIO()
                image.save(stream, "JPEG", quality=90)
                jpeg = stream.getvalue()
                box = color_box(jpeg, "blue")
                self.assertIsNotNone(box)  # Remains a navigation candidate.
                self.assertFalse(reportable_box(box))
                obs = replace(at(1_000_000_000), instruction="找蓝方块并确认",
                              rgb_jpeg={a: jpeg for a in AGENT_IDS})
                policy = RGBSearchPolicy()
                response = policy(obs)
                self.assertFalse(response.reports)
                self.assertEqual(policy.diagnostics["uav_0"]["phase"], "CENTER")
                self.assertTrue(policy.diagnostics["uav_0"]["visible"])
                self.assertFalse(policy.diagnostics["uav_0"]["report_ready"])
                self.assertEqual(response.action.uav_0.dx, 0)
                if yaw_sign:
                    self.assertGreater(yaw_sign * response.action.uav_0.dyaw, 0)
                self.assertFalse(policy.found)
                frames = {a: replace(frame, objects={
                    "probe_red": ObjectAnnotation(None), "probe_blue": ObjectAnnotation(None)})
                          for a, frame in private_frames(obs).items()}
                episode = SearchEpisode(manual_condition(obs.instruction, "blue"), obs.episode_id)
                episode.delivered(obs, frames, response, now_sim_time_ns=obs.timestamp_ns)
                self.assertIsNone(episode.evaluator.failure)
                # The fix is in the policy, not a relaxed evaluator: forcing
                # the old premature report must still fail independent truth.
                rejected = SearchEpisode(manual_condition(obs.instruction, "blue"), obs.episode_id)
                bad = GroundedAction(response.action, (TargetReport("uav_0", obs.timestamp_ns, box, 1., "observe"),))
                rejected.delivered(obs, frames, bad, now_sim_time_ns=obs.timestamp_ns)
                self.assertEqual(rejected.finish()["reason"], "ungrounded_target_report")

    def test_report_clearance_and_center_are_both_required(self):
        self.assertFalse(reportable_box(None))
        # Exact bbox from the failed blue step20: white material was clipped.
        self.assertFalse(reportable_box((226/256, 91/256, 252/256, 137/256)))
        self.assertFalse(reportable_box((.7, .4, .8, .5)))
        self.assertFalse(reportable_box((.45, .01, .55, .2)))
        self.assertFalse(reportable_box((.04, .4, .96, .6)))
        self.assertTrue(reportable_box((.45, .4, .55, .6)))

    def test_edge_candidate_then_centered_blue_reports_and_approaches(self):
        policy = RGBSearchPolicy()
        obs = replace(at(1_000_000_000), instruction="找蓝方块并确认")
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box",
                   return_value=(226/256, 91/256, 252/256, 137/256)):
            self.assertFalse(policy(obs).reports)
        # A subsequent fresh, fully framed view can report without an episode
        # reset. The receiver still waits, and found still needs real travel.
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box",
                   return_value=(.45, .4, .55, .6)):
            response = policy(replace(obs, timestamp_ns=1_500_000_000,
                                      image_timestamp_ns={a: 1_500_000_000 for a in AGENT_IDS}))
        self.assertEqual([r.event for r in response.reports], ["observe"])
        self.assertGreater(response.action.uav_0.dx, 0)
        self.assertEqual(response.action.uav_1.dx, 0)
        self.assertFalse(policy.found)

    def test_control_stamp_roundtrips_each_vehicles_actual_telemetry(self):
        from types import SimpleNamespace
        control = object.__new__(SearchFlightControl)
        control.use_telemetry_timestamps = True
        control.odometry = {0: SimpleNamespace(timestamp=100), 1: SimpleNamespace(timestamp=200)}
        self.assertEqual(control.stamp_us(0), 100)
        self.assertEqual(control.stamp_us(1), 200)
        # No elapsed host-time extrapolation, even with unchanged data.
        self.assertEqual(control.stamp_us(0), 100)
        control.odometry[0].timestamp = 500
        self.assertEqual(control.stamp_us(0), 500)
        control.odometry[1].timestamp = 0
        with self.assertRaises(ValueError):
            control.stamp_us(1)

    def test_old_scan_remains_yaw_only(self):
        action = RGBScanPolicy()(observation()).action
        for a in AGENT_IDS:
            self.assertEqual(getattr(action, a).vector()[:3], [0., 0., 0.])

    def test_pixels_control_turn_and_forward_not_private_answer(self):
        policy = RGBSearchPolicy()
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box", return_value=None):
            response = policy(at(1_000_000_000))
        self.assertLess(response.action.uav_0.dyaw, 0)
        self.assertEqual(response.action.uav_1.dx, 0)
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box", return_value=(.7, .4, .8, .5)):
            action = policy(at(1_500_000_000)).action.uav_0
        self.assertLess(action.dyaw, 0)  # Image right -> current-body right.
        self.assertEqual(action.dx, 0)
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box", return_value=(.45, .4, .55, .5)):
            action = policy(at(2_000_000_000)).action.uav_0
        self.assertGreater(action.dx, 0)

    def test_no_actual_motion_no_found_even_if_requested_repeatedly(self):
        policy = RGBSearchPolicy()
        with patch("benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box", return_value=(.45, .4, .55, .5)):
            for k in range(20):
                response = policy(at(1_000_000_000 + k * 500_000_000))
                self.assertFalse(any(r.event in {"found", "confirm"} for r in response.reports))

    def test_slow_simulation_needs_full_camera_time_dwell(self):
        policy = RGBSearchPolicy()
        module = "benchmark.px4_gazebo.model_handoff.search_motion_policy.color_box"
        with patch(module, return_value=(.45, .4, .55, .6)):
            policy(at(1_000_000_000))
        # Slow GUI: frequent wall-clock decisions advance only .2 simulated
        # seconds each. Seven fresh stable frames, not just three calls, are
        # needed to span 1.2 simulated seconds. Never shorten the evidence.
        with patch(module, return_value=(.44, .4, .56, .6)):
            for k in range(7):
                response = policy(at(1_200_000_000 + k * 200_000_000, scout=1.12))
                self.assertEqual(policy.diagnostics["uav_0"]["stable_frames"], k + 1)
                self.assertAlmostEqual(policy.diagnostics["uav_0"]["stable_dwell_s"], k * .2)
                self.assertEqual(policy.found, k == 6)
                self.assertEqual(any(r.event == "found" for r in response.reports), k == 6)
                self.assertEqual(response.action.uav_0.dx, 0.)

    def test_hold_transforms_ned_clockwise_yaw_to_current_flu(self):
        state = VehicleState((0., 0., 1.8), (0., 0., 0.), math.pi / 2, "ok")
        action = hold(state, (1., 0., 2.))
        self.assertAlmostEqual(action.dx, 0)
        self.assertGreater(action.dy, 0)
        self.assertGreater(action.dz, 0)

    def test_reset_stale_instruction_and_travel_guards(self):
        policy = RGBSearchPolicy()
        obs = at(1_000_000_000)
        policy(obs)
        with self.assertRaises(ValueError):
            policy(obs)
        with self.assertRaises(ValueError):
            policy(replace(at(1_500_000_000), instruction="找蓝球"))
        with self.assertRaises(RuntimeError):
            policy(at(2_000_000_000, scout=3.))
        response = policy(at(1_000_000_000, episode="new"))
        self.assertFalse(any(r.event == "confirm" for r in response.reports))

    def test_rotation_only_and_moving_away_rejected(self):
        for scout in (0., -.9):
            episode = gate_episode()
            obs = at(2_000_000_000, scout=scout)
            deliver(episode, obs, report(obs, "uav_0", "found"))
            self.assertEqual(episode.finish()["reason"], "found_without_actual_scout_approach")

    def test_receiver_must_move_after_found_not_before(self):
        episode = gate_episode()
        obs = at(2_000_000_000, scout=1., receiver=1.)
        deliver(episode, obs, report(obs, "uav_0", "found"))
        obs = at(3_000_000_000, scout=1., receiver=1.)
        deliver(episode, obs, report(obs, "uav_1", "confirm"))
        self.assertEqual(episode.finish()["reason"], "confirm_without_actual_receiver_approach_and_dwell")

    def test_fresh_approach_then_stable_private_views_complete(self):
        episode = gate_episode()
        obs = at(2_000_000_000, scout=1.)
        deliver(episode, obs, report(obs, "uav_0", "found"))
        # Centered private box supports dwell, independent of policy box.
        for k in range(3):
            obs = at(3_000_000_000 + k * 500_000_000, scout=1., receiver=1.)
            frames = private_frames(obs)
            frames["uav_1"] = replace(frames["uav_1"], objects={
                "probe_red": ObjectAnnotation((.4, .3, .6, .5)), "probe_blue": ObjectAnnotation(None)})
            r = TargetReport("uav_1", obs.timestamp_ns, (.4, .3, .6, .5), 1., "confirm" if k == 2 else "observe")
            episode.delivered(obs, frames, GroundedAction(ModelJointAction(), (r,)), now_sim_time_ns=obs.timestamp_ns)
        result = episode.finish()
        self.assertEqual(result["category"], "success")
        self.assertAlmostEqual(result["motion_metrics"]["receiver_range_reduction_m"], 1.)
        self.assertEqual(result["motion_calibration"]["receiver_xy_at_found"], (0., 0.))
        self.assertEqual(result["motion_calibration"]["private_local_goals"]["uav_0"], (4., 0., 2.4))

    def test_unstable_or_missing_views_cannot_dwell(self):
        for fault in ("fast", "invisible", "gap"):
            episode = gate_episode()
            obs = at(2_000_000_000, scout=1.)
            deliver(episode, obs, report(obs, "uav_0", "found"))
            for k in range(3):
                obs = at(3_000_000_000 + k * (1_000_000_000 if fault == "gap" else 500_000_000),
                         scout=1., receiver=1., speed=.3 if fault == "fast" else 0.)
                frames = private_frames(obs)
                box = None if fault == "invisible" else (.4, .3, .6, .5)
                frames["uav_1"] = replace(frames["uav_1"], objects={
                    "probe_red": ObjectAnnotation(box), "probe_blue": ObjectAnnotation(None)})
                r = TargetReport("uav_1", obs.timestamp_ns, (.4, .3, .6, .5), 1., "confirm" if k == 2 else "observe")
                episode.delivered(obs, frames, GroundedAction(ModelJointAction(), (r,)), now_sim_time_ns=obs.timestamp_ns)
            self.assertNotEqual(episode.finish()["category"], "success")

    def test_private_goal_calibration_respects_different_origins(self):
        tree = ET.ElementTree(ET.fromstring('<sdf><world><model name="x"><pose>4 1 2.4 0 0 0</pose></model></world></sdf>'))
        goals = private_local_goals(tree, "x", {0: 0., 1: 0.})
        self.assertEqual(goals["uav_0"][:2], (1., -4.))
        self.assertEqual(goals["uav_1"][:2], (4., -4.))

    def test_kinematic_render_loop_both_colors(self):
        # Pixels are rendered from independent virtual camera positions;
        # private geometry never enters the policy. No PX4 dynamics claimed.
        for color, goal in (("red", (4., 1.)), ("blue", (4., -1.))):
            policy = RGBSearchPolicy()
            coords = {"uav_0": [0., 0., 0.], "uav_1": [0., 0., 0.]}
            found = confirmed = False
            for k in range(115):
                obs = at(1_000_000_000 + k * 500_000_000)
                images, states = {}, {}
                for i, a in enumerate(AGENT_IDS):
                    x, y, angle = coords[a]
                    wx = -y  # Initial ENU yaw=pi/2; initial-FLU left=-world-X.
                    wy = x - 3 * i
                    bearing = math.atan2(goal[1] - wy, goal[0] - wx)
                    error = math.atan2(math.sin(bearing - (math.pi / 2 + angle)), math.cos(bearing - (math.pi / 2 + angle)))
                    distance = math.hypot(goal[0] - wx, goal[1] - wy)
                    image = Image.new("RGB", (256, 256), (80, 80, 80))
                    if abs(error) < .5:
                        center, width = 128 - 200 * math.tan(error), 200 * .8 / distance
                        ImageDraw.Draw(image).rectangle((center-width/2, 110, center+width/2, 110+width),
                                                       fill=(235, 4, 4) if color == "red" else (4, 4, 235))
                    stream = BytesIO()
                    image.save(stream, "JPEG", quality=90)
                    images[a] = stream.getvalue()
                    states[a] = VehicleState((x, y, 2.), (0., 0., 0.), -angle, "ok")
                obs = replace(obs, instruction="找红球并确认" if color == "red" else "找蓝方块并确认", rgb_jpeg=images, states=states)
                response = policy(obs)
                for r in response.reports:
                    found |= r.event == "found"
                    confirmed |= r.event == "confirm"
                if confirmed:
                    break
                for a in AGENT_IDS:
                    d, p = getattr(response.action, a), coords[a]
                    p[0] += d.dx * math.cos(p[2]) - d.dy * math.sin(p[2])
                    p[1] += d.dx * math.sin(p[2]) + d.dy * math.cos(p[2])
                    p[2] += d.dyaw
            self.assertTrue(found, color)
            self.assertTrue(confirmed, color)
            self.assertGreater(math.hypot(*coords["uav_0"][:2]), .8)
            self.assertGreater(math.hypot(*coords["uav_1"][:2]), .8)


if __name__ == "__main__":
    unittest.main()
