"""ROS-free parameterized long-horizon search/handoff/verify task."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping

from ...communication import CommunicationChannel
from ...contract import AGENT_IDS, Action, JointAction
from ...task_api import validate_policy_observation


@dataclass
class SearchHandoffTask:
    """A staged task family with search, allocation, handoff, and verification."""

    task_id: str = "dual_search_handoff_verify_v1"
    simulator: str = "mock"
    horizon: int = 160
    control_hz: float = 10.0
    confirmation_radius: float = 0.8
    collision_distance: float = 0.6

    def __init__(self, task_spec: Mapping[str, Any] | None = None) -> None:
        spec = dict(task_spec or {})
        self.task_id = str(spec.get("task_id", self.task_id))
        self.horizon = int(spec.get("episode_horizon", self.horizon))
        self.control_hz = float(spec.get("control_hz", self.control_hz))
        self.communication = None

    def reset(self, seed: int, task_spec: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        rng = random.Random(seed)
        self.seed = seed
        self.step_index = 0
        self.target_position = (5.0 + rng.randint(0, 3), 0.0, 0.0)
        self.positions = {"uav_0": [0.0, 0.0, 0.0], "uav_1": [0.0, 3.0, 0.0]}
        self.phase = "search"
        self.failed = False
        self.events: list[dict[str, Any]] = []
        self.delivered_messages: list[dict[str, Any]] = []
        self.sent_messages = 0
        self.communication = CommunicationChannel(
            bandwidth_bytes_per_second=int((task_spec or {}).get("bandwidth_bytes_per_second", 256)),
            latency_ms=float((task_spec or {}).get("latency_ms", 0)),
            drop_probability=float((task_spec or {}).get("drop_probability", 0)),
            control_hz=self.control_hz,
            seed=seed,
        )
        observation = self._observation()
        validate_policy_observation(observation)
        return observation

    def step(self, action: JointAction | Mapping[str, Any]):
        if isinstance(action, Mapping):
            action = JointAction(**action)
        self.step_index += 1
        self._move("uav_0", action.uav_0)
        self._move("uav_1", action.uav_1)
        if action.message is not None:
            self.sent_messages += 1
            message = self.communication.send(sender="uav_0", receiver="uav_1", payload=action.message, t=self.step_index)
            if not message.dropped:
                self.events.append({"event_type": "message_sent", "step": self.step_index, "success": True})
        for message in self.communication.receive(receiver="uav_1", t=self.step_index):
            self.delivered_messages.append({"payload": message.payload, "step": self.step_index})
            if message.payload.get("intent") == "found" and self._near("uav_0") and self.phase == "search":
                self.phase = "pre_position"
                self.events.append({"event_type": "candidate_found", "step": self.step_index, "success": True})
        if self.phase == "pre_position" and self._near("uav_1"):
            self.phase = "handoff"
            self.events.append({"event_type": "handoff", "step": self.step_index, "success": True})
        if self.phase == "handoff" and self._near("uav_1"):
            self.phase = "verify"
        if self.phase == "verify" and action.message and action.message.get("intent") == "confirm" and self._near("uav_1"):
            self.phase = "complete"
            self.events.append({"event_type": "verify", "step": self.step_index, "success": True})
        if self._distance(self.positions["uav_0"], self.positions["uav_1"]) < self.collision_distance:
            self.failed = True
            self.events.append({"event_type": "collision", "step": self.step_index, "success": False})
        timeout = self.step_index >= self.horizon and self.phase != "complete"
        if timeout:
            self.events.append({"event_type": "timeout", "step": self.step_index, "success": False})
        done = self.phase == "complete" or self.failed or timeout
        observation = self._observation()
        validate_policy_observation(observation)
        return observation, float(self.phase == "complete") - float(self.failed), done, {"truth": self._truth()}

    def episode_record(self) -> Mapping[str, Any]:
        success = self.phase == "complete" and not self.failed
        return {
            "schema_version": "0.2",
            "episode_id": f"{self.task_id}__seed_{self.seed}",
            "task_id": self.task_id,
            "split": "source_train",
            "seed": self.seed,
            "agent_ids": list(AGENT_IDS),
            "environment": {"simulator": self.simulator, "version": "task-family-mock-0.1", "map_id": "generated", "reset_hash": self._reset_hash(), "control_hz": self.control_hz},
            "steps": [],
            "outcome": {"terminated": True, "termination_reason": "success" if success else ("collision" if self.failed else "timeout"), "success": success, "metrics": {"task_success": float(success), "handoff_success": float(any(e["event_type"] == "handoff" for e in self.events)), "confirmation_success": float(self.phase == "complete"), "communication_bytes": float(self.communication.bytes_sent), "collision_rate": float(self.failed), "completion_steps": float(self.step_index)}},
            "events": list(self.events),
            "truth": self._truth(),
        }

    def _move(self, agent_id: str, action: Action) -> None:
        command = action.clipped()
        dt = 1.0 / self.control_hz
        self.positions[agent_id][0] += command.vx * dt
        self.positions[agent_id][1] += command.vy * dt
        self.positions[agent_id][2] += command.vz * dt

    def _near(self, agent_id: str) -> bool:
        return self._distance(self.positions[agent_id], self.target_position) <= self.confirmation_radius

    def _observation(self) -> dict[str, Any]:
        return {"task": {"task_id": self.task_id, "instruction": "Search, hand off, and verify the target.", "phase": self.phase}, "timestamp_ms": int(self.step_index * 1000 / self.control_hz), "observations": {agent_id: {"agent_id": agent_id, "rgb_uri": f"mock://{agent_id}/frame_{self.step_index:06d}.jpg", "own_state": {"position": list(self.positions[agent_id]), "velocity": [0.0, 0.0, 0.0], "health": "ok"}, "last_message": self.delivered_messages[-1]["payload"] if self.delivered_messages else None} for agent_id in AGENT_IDS}}

    def _truth(self) -> dict[str, Any]:
        return {"target_position": list(self.target_position), "agent_positions": {key: list(value) for key, value in self.positions.items()}, "phase": self.phase}

    def _reset_hash(self) -> str:
        return hashlib.sha256(f"{self.task_id}:{self.seed}:{self.target_position}".encode()).hexdigest()

    @staticmethod
    def _distance(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5
