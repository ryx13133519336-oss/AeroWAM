"""Pilot integration/negative controls. No flight capability claims."""

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from .grounded_interface import GroundedAction
from .language_evaluator import ObjectAnnotation, ObservedFrame
from .protocol import AGENT_IDS, ModelJointAction, ModelObservation, VehicleState, WaypointDelta
from .search_episode import SearchEpisode
from .search_flight_control import SearchFlightControl, public_state, safety_problem, px4_topic
from .search_pilot_scene import manual_condition, paired_conditions
from .search_reporter import GroundedCheckpointPolicy, PilotReporter, color_box, requested_color
from .search_pilot import agent_major_from_dependencies, resolve_run_duration, run_progress
from .teammate_response import ResponseSpec, TeammateResponse, request_control, audit_response_trace

RED_BOX, BLUE_BOX = (40 / 256, 60 / 256, 91 / 256, 111 / 256), (160 / 256, 80 / 256, 211 / 256, 131 / 256)


def jpeg():
    image = Image.new("RGB", (256, 256), (80, 80, 80))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 60, 90, 110), fill=(235, 4, 4))
    draw.rectangle((160, 80, 210, 130), fill=(4, 4, 235))
    stream = BytesIO()
    image.save(stream, "JPEG", quality=90)
    return stream.getvalue()


def observation(text="请找到红色球，并让另一架无人机确认。", stamp=1_000_000_000, episode="opaque"):
    return ModelObservation(episode, text, stamp, {a: stamp for a in AGENT_IDS}, {a: jpeg() for a in AGENT_IDS},
                            {a: VehicleState((0., 0., 2.), (0., 0., 0.), 0., "ok") for a in AGENT_IDS})


def private_frames(obs):
    return {a: ObservedFrame(a, obs.image_timestamp_ns[a], obs.image_timestamp_ns[a],
                             {"probe_red": ObjectAnnotation(RED_BOX), "probe_blue": ObjectAnnotation(BLUE_BOX)})
            for a in AGENT_IDS}


def complete(private_target, text):
    episode = SearchEpisode(manual_condition(text, private_target), "opaque")
    reporter = PilotReporter()
    for stamp in (1_000_000_000, 1_500_000_000):
        obs = observation(text, stamp)
        response = GroundedAction(ModelJointAction(), reporter.reports(obs))
        episode.delivered(obs, private_frames(obs), response, now_sim_time_ns=stamp)
    return episode.finish()


class SearchPilotTests(unittest.TestCase):
    def test_run_duration_wall_budget_is_bounded_and_explicit(self):
        self.assertEqual(resolve_run_duration("rgb_search"), 90.)
        self.assertEqual(resolve_run_duration("rgb_search", 60.), 60.)
        self.assertEqual(resolve_run_duration("rgb_search", 120.), 120.)
        for mode in ("rgb_scan", "checkpoint"):
            self.assertEqual(resolve_run_duration(mode), 30.)
            self.assertEqual(resolve_run_duration(mode, 60.), 60.)
            with self.assertRaises(ValueError):
                resolve_run_duration(mode, 90.)
        for value in (0, 4.9, 120.01, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                resolve_run_duration("rgb_search", value)

    def test_progress_does_not_turn_partial_evidence_into_success(self):
        obs = observation()
        episode = SearchEpisode(manual_condition(obs.instruction, "red"), obs.episode_id)
        episode.evaluator.found = True
        # Approaching / stabilizing is not a confirm event. Diagnostics must
        # never mutate acceptance or fabricate completion at the deadline.
        gate = SimpleNamespace(metrics={"receiver_stable_dwell_s": .496})
        episode.motion_gate = gate
        policy = SimpleNamespace(diagnostics={"uav_1": {"phase": "STABILIZE"}})
        progress = run_progress(episode, policy)
        self.assertTrue(progress["found_accepted"])
        self.assertFalse(progress["confirm_accepted"])
        self.assertEqual(progress["motion_metrics"]["receiver_stable_dwell_s"], .496)
        progress["motion_metrics"].clear()
        self.assertEqual(gate.metrics["receiver_stable_dwell_s"], .496)
        episode.motion_gate = None
        self.assertEqual(episode.finish()["reason"], "incomplete_task_evidence")

    def test_installed_agent_runtime_not_source_checkout_version(self):
        self.assertEqual(agent_major_from_dependencies("libmicroxrcedds_agent.so.3.0 => /usr/local/lib/x"), 3)
        self.assertEqual(agent_major_from_dependencies("libmicroxrcedds_agent.so.2.4 => /tmp/x"), 2)
        self.assertIsNone(agent_major_from_dependencies("statically linked"))

    def test_px4_versioned_ack_topic(self):
        self.assertEqual(px4_topic(1, "out", "vehicle_command_ack", SimpleNamespace(MESSAGE_VERSION=1)),
                         "/annotation_smoke_1/fmu/out/vehicle_command_ack_v1")
        self.assertEqual(px4_topic(0, "in", "vehicle_command", SimpleNamespace(MESSAGE_VERSION=0)),
                         "/annotation_smoke_0/fmu/in/vehicle_command")

    def test_manual_text_exact_and_separate_private_answer(self):
        text = "请让两架无人机寻找蓝色的方块，再相互确认。"
        task = manual_condition(text, "blue")
        self.assertEqual(task.definition.instruction, text)
        self.assertEqual(task.target_id, "probe_blue")
        # Deliberately inconsistent human annotation stays independent of parser.
        other = manual_condition(text, "red")
        self.assertNotEqual(task.target_id, other.target_id)
        self.assertNotIn("target_id", task.definition.public_context())

    def test_pair_order_reproducible_answer_changed(self):
        pair = paired_conditions(7)
        self.assertEqual(pair, paired_conditions(7))
        self.assertEqual({b.target_id for b in pair}, {"probe_red", "probe_blue"})
        self.assertEqual(len({b.definition.instruction for b in pair}), 2)

    def test_rgb_color_boxes_no_private_labels(self):
        self.assertIsNotNone(color_box(jpeg(), "red"))
        self.assertIsNotNone(color_box(jpeg(), "blue"))
        self.assertLess(color_box(jpeg(), "red")[2], color_box(jpeg(), "blue")[0])

    def test_limited_reporter_language_refuses_ambiguous_negated(self):
        self.assertEqual(requested_color("找到蓝方块并确认"), "blue")
        self.assertEqual(requested_color("Find the RED sphere."), "red")
        for text in ("找红球或者蓝方块", "找那个目标", "不要找红球", "Do not find the blue cube"):
            with self.assertRaises(ValueError):
                requested_color(text)

    def test_learned_actions_and_raw_manual_text_preserved(self):
        calls = []
        action = ModelJointAction(uav_0=WaypointDelta(.2, -.1, 0., .03), intent="handoff")
        def action_policy(wire):
            calls.append(wire)
            self.assertNotIn("target_id", wire)
            self.assertNotIn("instance_labels", wire)
            return action.to_wire()
        obs = observation()
        response = GroundedCheckpointPolicy(action_policy)(obs)
        self.assertEqual(response.action, action)
        self.assertEqual(calls[0]["instruction"], obs.instruction)

    def test_actual_rgb_reports_pass_both_independent_answers(self):
        self.assertEqual(complete("red", "寻找红球并双机确认")["category"], "success")
        self.assertEqual(complete("blue", "寻找蓝色方块并双机确认")["category"], "success")

    def test_fixed_wrong_target_cannot_pass_changed_instruction(self):
        result = complete("blue", "寻找红球并双机确认")
        self.assertEqual((result["category"], result["reason"]), ("model_failure", "wrong_instruction_target"))

    def test_reporter_history_reset_between_manual_episodes(self):
        reporter = PilotReporter()
        self.assertEqual([r.event for r in reporter.reports(observation())], ["found", "observe"])
        self.assertEqual([r.event for r in reporter.reports(observation(stamp=1_500_000_000))], ["observe", "confirm"])
        other = observation("找蓝色方块并确认", 2_000_000_000, "fresh_uuid")
        self.assertEqual([r.event for r in reporter.reports(other)], ["found", "observe"])

    def test_no_fabricated_delivery_or_reused_frames(self):
        obs = observation()
        episode = SearchEpisode(manual_condition(obs.instruction, "red"), obs.episode_id)
        frames = private_frames(obs)
        episode.validate_capture(obs, frames)
        self.assertFalse(episode.evaluator.frames)  # Preparing isn't delivery.
        response = GroundedAction(ModelJointAction(), PilotReporter().reports(obs))
        episode.delivered(obs, frames, response, now_sim_time_ns=obs.timestamp_ns)
        with self.assertRaises(ValueError):
            episode.delivered(obs, frames, response, now_sim_time_ns=obs.timestamp_ns)
        with self.assertRaises(ValueError):
            episode.validate_capture(replace(obs, episode_id="wrong"), frames)

    def test_late_safety_and_infra_classification(self):
        obs = observation()
        for method, expected in (("model_error", "model_failure"), ("observation_error", "simulator_error"),
                                 ("annotation_error", "evaluator_error")):
            episode = SearchEpisode(manual_condition(obs.instruction, "red"), obs.episode_id)
            getattr(episode, method)("injected")
            self.assertEqual(episode.finish()["category"], expected)

    def test_public_own_state_uses_local_initial_frame(self):
        msg = SimpleNamespace(position=(2., 3., -2.), velocity=(1., 0., 0.), q=(1., 0., 0., 0.))
        state = public_state(msg, (1., 3., 0.), 0.)
        self.assertEqual(state.position, (1., 0., 2.))
        self.assertEqual(state.velocity, (1., 0., 0.))

    def test_receive_watchdog_and_minimum_separation(self):
        msg = SimpleNamespace(position=(0., 0., -2.), velocity=(0., 0., 0.), q=(1., 0., 0., 0.))
        odom = {0: msg, 1: msg}
        status = {i: SimpleNamespace(failsafe=False) for i in (0, 1)}
        rec = {(kind, i): 10. for kind in ("status", "odometry") for i in (0, 1)}
        origins = {i: (0., 0., 0.) for i in (0, 1)}
        self.assertIsNone(safety_problem(odom, status, rec, origins, 10.1))
        self.assertEqual(safety_problem(odom, status, rec, origins, 10.6)[0], "simulator_error")
        odom[1] = SimpleNamespace(position=(3., 0., -2.), velocity=msg.velocity, q=msg.q)
        self.assertEqual(safety_problem(odom, status, rec, origins, 10.1)[1], "minimum_separation")

    def test_control_takeoff_action_conversion_and_landing_commands(self):
        class Publisher:
            def __init__(self):
                self.messages = []
            def publish(self, message):
                self.messages.append(message)
        class ROS:
            def create_publisher(self, *args):
                return Publisher()
            def create_timer(self, period, callback):
                return callback
            def get_clock(self):
                return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
        fake = SimpleNamespace(OffboardControlMode=SimpleNamespace, TrajectorySetpoint=SimpleNamespace,
                               VehicleCommand=SimpleNamespace)
        odom = {i: SimpleNamespace(position=[0., 0., 0.], velocity=[0., 0., 0.], q=[1., 0., 0., 0.]) for i in (0, 1)}
        status = {i: SimpleNamespace(arming_state=1, nav_state=0, failsafe=False) for i in (0, 1)}
        rec = {(kind, i): 10. for kind in ("status", "odometry") for i in (0, 1)}
        with patch.dict("sys.modules", {"px4_msgs": SimpleNamespace(msg=fake), "px4_msgs.msg": fake}), \
                patch("benchmark.px4_gazebo.model_handoff.search_flight_control.control_qos", return_value="explicit_px4_qos"), \
                patch("benchmark.px4_gazebo.model_handoff.search_flight_control.time.monotonic", return_value=10.) as clock:
            link = SimpleNamespace(verified=True,
                                   reader_counts=lambda: {f"input_{i}": 1 for i in range(6)},
                                   publishers={i: tuple(Publisher() for _ in range(3)) for i in (0, 1)})
            control = SearchFlightControl(ROS(), odom, status, rec, link)
            self.assertIs(control.publishers, link.publishers)
            control.tick()
            self.assertEqual(control.commands, 0)  # Prestream precedes mode/arm.
            clock.return_value = 12.1
            rec.update({key: 12.1 for key in rec})
            control.tick()
            self.assertEqual(control.publishers[0][2].messages[-1].command, 176)
            self.assertEqual(control.publishers[0][2].messages[-1].target_system, 21)
            for i in (0, 1):
                status[i].nav_state = 14
                status[i].arming_state = 2
                odom[i].position[2] = -2.
            clock.return_value = 12.15
            rec.update({key: 12.15 for key in rec})
            response = TeammateResponse(ResponseSpec("delayed"))
            trace = []
            request = ModelJointAction(uav_0=WaypointDelta(.5, .5, .2, .1), uav_1=WaypointDelta(.3))
            request_control(control, response, request, sim_stamp_ns=1_000_000_000, trace=trace)
            control.tick()
            self.assertEqual(control.phase, "RUN")
            point = control.publishers[0][1].messages[-1]
            self.assertGreater(point.velocity[0], 0)
            self.assertLess(point.velocity[1], 0)  # FLU left -> NED west.
            self.assertLess(point.velocity[2], 0)  # FLU up -> NED negative.
            self.assertAlmostEqual(point.yawspeed, -.2)
            self.assertEqual(control.publishers[1][1].messages[-1].velocity, [0., 0., 0.])
            clock.return_value = 12.2
            rec.update({key: 12.2 for key in rec})
            request_control(control, response, request, sim_stamp_ns=1_750_000_000, trace=trace)
            control.tick()
            self.assertGreater(control.publishers[1][1].messages[-1].velocity[0], 0.)
            self.assertTrue(response.pending)  # Pending response must not delay a safety landing.
            self.assertTrue(audit_response_trace(response.spec, trace)["trace_valid"])
            # The actual checked writers must remain matched during flight.
            link.reader_counts = lambda: {f"input_{i}": 0 for i in range(6)}
            clock.return_value = 12.25
            control.tick()
            self.assertEqual(control.failure, ("simulator_error", "control_link_reader_lost"))
            self.assertEqual(control.phase, "LANDING")
            self.assertEqual(control.publishers[0][2].messages[-1].command, 21)
            self.assertEqual(control.publishers[1][2].messages[-1].command, 21)
            self.assertFalse(any(m.command == 400 and m.param1 == 0. for m in control.publishers[0][2].messages))
            for state in status.values():
                state.arming_state = 1
            control.tick()
            self.assertEqual(control.phase, "DONE")


if __name__ == "__main__":
    unittest.main()
