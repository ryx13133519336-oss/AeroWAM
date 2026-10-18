"""No-arm gate for the isolated search pilot's *actual* control writers.

Matching is discovery evidence, not evidence of setpoint execution. A fresh
accepted disarm ACK checks command roundtrip while both vehicles are disarmed.
No mode, arm, force-disarm or motion command is sent by this gate.
"""

import time

from .search_flight_control import control_qos, px4_topic
from .annotation_session_smoke import INSTANCES


# All pilot participants live inside one WSL host. Keep this opt-in until
# real no-arm checks pass; never modify the user's network/global DDS profile.
LOOPBACK_PROFILE = """<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors><transport_descriptor><transport_id>search_loopback</transport_id>
    <type>UDPv4</type><interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>
  </transport_descriptor></transport_descriptors>
  <participant profile_name="search_loopback" is_default_profile="true"><rtps>
    <builtin>
      <initialPeersList><locator><udpv4><address>127.0.0.1</address></udpv4></locator></initialPeersList>
      <metatrafficUnicastLocatorList><locator><udpv4><address>127.0.0.1</address></udpv4></locator></metatrafficUnicastLocatorList>
    </builtin>
    <userTransports><transport_id>search_loopback</transport_id></userTransports>
    <useBuiltinTransports>false</useBuiltinTransports>
  </rtps></participant>
</profiles>
"""


class PX4ControlLink:
    def __init__(self, ros, timeout_s=25.):
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.ros, self.command_type = ros, VehicleCommand
        self.timeout_s = timeout_s
        self.publishers, self.topics = {}, {}
        for i in (0, 1):
            topics = [px4_topic(i, "in", name, kind) for name, kind in (
                ("offboard_control_mode", OffboardControlMode),
                ("trajectory_setpoint", TrajectorySetpoint), ("vehicle_command", VehicleCommand))]
            self.topics[i] = topics
            self.publishers[i] = tuple(ros.create_publisher(kind, topic, control_qos())
                                       for kind, topic in zip((OffboardControlMode, TrajectorySetpoint, VehicleCommand), topics))
        self.phase, self.reason, self.verified = "NOT_STARTED", None, False
        self.started = self.matched_since = None
        self.first_request_ns, self.last_request = {}, -float("inf")
        self.commands, self.history = [], []
        self.last_record = -float("inf")

    def reader_counts(self):
        return {self.topics[i][j]: p.get_subscription_count()
                for i, publishers in self.publishers.items() for j, p in enumerate(publishers)}

    def start(self):
        if self.phase != "NOT_STARTED":
            raise RuntimeError("Control link gate may only start once")
        self.started, self.phase = time.monotonic(), "WAIT_MATCH"

    def fail(self, reason):
        self.phase, self.reason, self.verified = "FAILED", reason, False

    def poll(self, acks, status, received):
        if self.phase in {"NOT_STARTED", "FAILED", "VERIFIED"}:
            return self.verified
        now = time.monotonic()
        counts = self.reader_counts()
        # A discovered second reader could be a foreign controller/session.
        # This isolated diagnostic expects exactly one PX4 reader per topic.
        if any(n > 1 for n in counts.values()):
            self.fail("control_link_unexpected_readers")
        elif any(i not in status or now - received.get(("status", i), -float("inf")) > 2.
                 for i in (0, 1)):
            self.fail("control_link_stale_status")
        elif any(status[i].arming_state != 1 or status[i].failsafe for i in (0, 1)):
            self.fail("control_link_requires_disarmed_healthy_vehicles")
        elif now - self.started >= self.timeout_s:
            self.fail("control_link_match_timeout" if self.phase == "WAIT_MATCH" else "control_link_ack_timeout")
        elif not all(n == 1 for n in counts.values()):
            if self.phase == "WAIT_ACK":
                self.fail("control_link_reader_lost")
            self.matched_since = None
        else:
            if self.matched_since is None:
                self.matched_since = now
            if now - self.matched_since >= .25:
                self.phase = "WAIT_ACK"
                if now - self.last_request >= 1.:
                    for i in (0, 1):
                        msg = self.command_type()
                        msg.timestamp = self.ros.get_clock().now().nanoseconds // 1000
                        msg.command, msg.param1, msg.param2 = 400, 0., 0.
                        msg.target_system, msg.target_component = INSTANCES[i] + 1, 1
                        msg.source_system, msg.source_component, msg.from_external = 255, 1, True
                        sent = time.monotonic_ns()
                        self.first_request_ns.setdefault(f"uav_{i}", sent)
                        self.publishers[i][2].publish(msg)
                        self.commands.append({"agent": f"uav_{i}", "command": 400, "param1": 0.,
                                              "param2": 0., "monotonic_ns": sent})
                    self.last_request = now
                accepted = {row["agent"] for row in acks
                            if row.get("agent") in self.first_request_ns and row.get("command") == 400
                            and row.get("result") == 0 and row.get("target_system") == 255
                            and row.get("target_component") == 1
                            and row.get("monotonic_ns", -1) >= self.first_request_ns[row["agent"]]}
                if accepted == {"uav_0", "uav_1"}:
                    self.phase, self.verified = "VERIFIED", True
        if now - self.last_record >= 1. or self.phase in {"FAILED", "VERIFIED"}:
            self.history.append({"elapsed_s": now - self.started, "phase": self.phase, "readers": counts,
                                 "reason": self.reason})
            self.last_record = now
        return self.verified

    def diagnostics(self):
        endpoints = {}
        for topics in self.topics.values():
            for topic in topics:
                try:
                    endpoints[topic] = [{"node": item.node_name, "namespace": item.node_namespace,
                                         "type": item.topic_type, "gid": list(item.endpoint_gid),
                                         "reliability": str(item.qos_profile.reliability),
                                         "durability": str(item.qos_profile.durability)}
                                        for item in self.ros.get_subscriptions_info_by_topic(topic)]
                except Exception as exc:
                    endpoints[topic] = {"diagnostic_error": f"{type(exc).__name__}:{exc}"}
        return {"schema": "aerowam-search-control-link/1", "phase": self.phase, "reason": self.reason,
                "verified": self.verified, "reader_counts": self.reader_counts(), "endpoints": endpoints,
                "history": self.history, "preflight_commands": self.commands,
                "setpoint_execution_verified": False}
