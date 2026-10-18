"""Fail-closed match/ACK gate tests, without ROS, motion or simulator."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .search_control_link import PX4ControlLink
from .search_flight_control import SearchFlightControl, control_qos


class Publisher:
    def __init__(self):
        self.readers, self.messages = 1, []

    def get_subscription_count(self):
        return self.readers

    def publish(self, msg):
        self.messages.append(msg)


class ROS:
    def create_publisher(self, *args):
        return Publisher()

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1000))

    def get_subscriptions_info_by_topic(self, topic):
        return []


class ControlLinkTests(unittest.TestCase):
    def test_control_writer_qos_is_live_not_cached_telemetry(self):
        qos = SimpleNamespace(
            QoSProfile=SimpleNamespace,
            ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable"),
            DurabilityPolicy=SimpleNamespace(VOLATILE="volatile"),
            HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"))
        with patch.dict("sys.modules", {"rclpy.qos": qos}):
            profile = control_qos()
        self.assertEqual(profile.reliability, "reliable")
        self.assertEqual(profile.durability, "volatile")
        self.assertEqual(profile.history, "keep_last")
        self.assertEqual(profile.depth, 10)

    def setUp(self):
        fake = SimpleNamespace(OffboardControlMode=SimpleNamespace, TrajectorySetpoint=SimpleNamespace,
                               VehicleCommand=SimpleNamespace)
        patches = [patch.dict("sys.modules", {"px4_msgs": SimpleNamespace(msg=fake), "px4_msgs.msg": fake}),
                   patch("benchmark.px4_gazebo.model_handoff.search_control_link.control_qos", return_value="px4"),
                   patch("benchmark.px4_gazebo.model_handoff.search_control_link.time.monotonic", return_value=10.),
                   patch("benchmark.px4_gazebo.model_handoff.search_control_link.time.monotonic_ns", return_value=10_000_000_000)]
        active = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.clock, self.clock_ns = active[2:]
        self.link = PX4ControlLink(ROS(), timeout_s=5.)
        self.status = {i: SimpleNamespace(arming_state=1, failsafe=False) for i in (0, 1)}
        self.received = {("status", i): 10. for i in (0, 1)}
        self.link.start()

    def poll(self, now, acks=()):
        self.clock.return_value, self.clock_ns.return_value = now, int(now * 1e9)
        self.received.update({key: now for key in self.received})
        return self.link.poll(acks, self.status, self.received)

    def acks(self, **changes):
        return [{"agent": f"uav_{i}", "command": 400, "result": 0, "target_system": 255,
                 "target_component": 1, "monotonic_ns": 10_400_000_000, **changes} for i in (0, 1)]

    def begin_probe(self):
        self.poll(10.)
        self.poll(10.3)
        self.assertEqual(self.link.phase, "WAIT_ACK")

    def test_all_six_writers_required_before_any_command(self):
        self.link.publishers[1][0].readers = 0
        self.poll(10.)
        self.poll(14.)
        self.assertFalse(self.link.commands)
        self.poll(15.1)
        self.assertEqual(self.link.reason, "control_link_match_timeout")
        self.assertFalse(self.link.verified)

    def test_two_fresh_accepted_correctly_addressed_acks(self):
        self.begin_probe()
        self.assertFalse(self.poll(10.4, self.acks()[:1]))
        self.assertTrue(self.poll(10.5, self.acks()))
        self.assertEqual(self.link.phase, "VERIFIED")
        for publishers in self.link.publishers.values():
            self.assertFalse(publishers[0].messages)  # No offboard-mode message.
            self.assertFalse(publishers[1].messages)  # No motion setpoint.
            self.assertTrue(publishers[2].messages)
            for msg in publishers[2].messages:
                self.assertEqual((msg.command, msg.param1, msg.param2), (400, 0., 0.))

    def test_stale_wrong_address_or_rejected_ack_cannot_unlock_gate(self):
        self.begin_probe()
        for changes in ({"monotonic_ns": 9}, {"target_system": 21}, {"target_component": 2},
                        {"result": 2}, {"command": 176}):
            self.assertFalse(self.poll(10.4, self.acks(**changes)))
        self.poll(15.1)
        self.assertEqual(self.link.reason, "control_link_ack_timeout")

    def test_reader_loss_and_duplicate_readers_fail_closed(self):
        self.begin_probe()
        self.link.publishers[0][1].readers = 0
        self.poll(10.4, self.acks())
        self.assertEqual(self.link.reason, "control_link_reader_lost")
        self.assertFalse(self.link.verified)

    def test_unexpected_reader_rejected_before_probe(self):
        self.link.publishers[0][2].readers = 2
        self.poll(10.)
        self.assertEqual(self.link.reason, "control_link_unexpected_readers")
        self.assertFalse(self.link.commands)

    def test_stale_or_armed_status_prevents_probe(self):
        self.received[("status", 0)] = 1.
        self.link.poll([], self.status, self.received)
        self.assertEqual(self.link.reason, "control_link_stale_status")
        self.assertFalse(self.link.commands)

    def test_arm_state_rejected_before_probe(self):
        self.status[1].arming_state = 2
        self.poll(10.)
        self.assertEqual(self.link.reason, "control_link_requires_disarmed_healthy_vehicles")
        self.assertFalse(self.link.commands)

    def test_flight_controller_cannot_bypass_gate(self):
        with self.assertRaisesRegex(RuntimeError, "Verified control link"):
            SearchFlightControl(ROS(), {}, {}, {}, self.link)
        self.assertFalse(self.link.commands)

    def test_diagnostics_have_all_topics_and_failure_history(self):
        self.link.publishers[0][2].readers = 0
        self.poll(10.)
        self.poll(15.1)
        result = self.link.diagnostics()
        self.assertEqual(len(result["reader_counts"]), 6)
        self.assertEqual(len(result["endpoints"]), 6)
        self.assertEqual(result["history"][-1]["reason"], "control_link_match_timeout")
        self.assertFalse(result["setpoint_execution_verified"])


if __name__ == "__main__":
    unittest.main()
