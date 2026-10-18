"""Receive-time sampling gate for opt-in traces, before stateful decisions."""

from dataclasses import dataclass
from typing import Mapping

from .protocol import AGENT_IDS

SAMPLING_PROFILE = "aerowam-dual-frame-receive-sampling/1-draft"
MAX_RECEIVE_AGE_NS = 250_000_000
MAX_PAIR_SKEW_NS = 100_000_000


@dataclass
class TraceSamplingGate:
    timeout_ns: int
    waiting_since_ns: int | None = None

    def check(self, *, now_ns: int, stamps: Mapping[str, int], previous_stamps: Mapping[str, int],
              image_received_ns: Mapping[str, int], image_advanced_ns: Mapping[str, int],
              odometry_received_ns: Mapping[str, int]) -> tuple[bool, str | None]:
        reason = None
        for agent in AGENT_IDS:
            if stamps[agent] <= previous_stamps[agent]:
                reason = f"{agent} image stamp has not advanced"
                break
            for kind, received in (("image", image_received_ns[agent]),
                                   ("image stamp advance", image_advanced_ns[agent]),
                                   ("odometry", odometry_received_ns[agent])):
                if not 0 <= now_ns - received <= MAX_RECEIVE_AGE_NS:
                    reason = f"{agent} {kind} receive age outside sampling window"
                    break
            if reason:
                break
        if reason is None and abs(stamps["uav_0"] - stamps["uav_1"]) > MAX_PAIR_SKEW_NS:
            reason = "camera pair skew above 100 ms"
        if reason is None:
            self.waiting_since_ns = None
            return True, None
        if self.waiting_since_ns is None:
            self.waiting_since_ns = now_ns
        if now_ns - self.waiting_since_ns >= self.timeout_ns:
            return False, f"trace sampling timeout: {reason}"
        return False, None
