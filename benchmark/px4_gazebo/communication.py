"""Deterministic bandwidth, latency, and packet-drop simulation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any


@dataclass(frozen=True)
class DeliveredMessage:
    sender: str
    receiver: str
    payload: Any
    sent_t: int
    deliver_t: int | None
    bytes: int
    dropped: bool


class CommunicationChannel:
    def __init__(self, *, bandwidth_bytes_per_second: int = 64, latency_ms: float = 100.0, drop_probability: float = 0.1, control_hz: float = 10.0, seed: int = 0) -> None:
        if bandwidth_bytes_per_second < 0 or latency_ms < 0 or not 0 <= drop_probability <= 1:
            raise ValueError("invalid communication configuration")
        self.bandwidth = int(bandwidth_bytes_per_second)
        self.control_hz = float(control_hz)
        self.latency_steps = int(round(latency_ms * control_hz / 1000.0))
        self.drop_probability = float(drop_probability)
        self._rng = random.Random(seed)
        self._pending: list[DeliveredMessage] = []
        self.bytes_sent = 0
        self.dropped = 0
        self._byte_credit = 0.0
        self._last_send_t = 0

    @staticmethod
    def payload_size(payload: Any) -> int:
        return len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))

    def send(self, *, sender: str, receiver: str, payload: Any, t: int) -> DeliveredMessage:
        size = self.payload_size(payload)
        elapsed_steps = max(1, int(t) - self._last_send_t + 1)
        self._byte_credit = min(float(self.bandwidth), self._byte_credit + self.bandwidth * elapsed_steps / self.control_hz)
        self._last_send_t = int(t)
        over_budget = size > self._byte_credit
        dropped = self.bandwidth <= 0 or over_budget or self._rng.random() < self.drop_probability
        message = DeliveredMessage(sender, receiver, payload, int(t), None if dropped else int(t + self.latency_steps), size, dropped)
        if dropped:
            self.dropped += 1
        else:
            self._byte_credit -= size
            self.bytes_sent += size
            self._pending.append(message)
        return message

    def receive(self, *, receiver: str, t: int) -> list[DeliveredMessage]:
        ready = [item for item in self._pending if item.receiver in {receiver, "broadcast"} and item.deliver_t is not None and item.deliver_t <= t]
        self._pending = [item for item in self._pending if item not in ready]
        return ready
