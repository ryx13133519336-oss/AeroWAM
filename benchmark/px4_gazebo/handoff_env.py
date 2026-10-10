"""Deterministic handoff fixture with the same contract used by the PX4 adapter."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Mapping

from .communication import CommunicationChannel, DeliveredMessage
from .contract import AGENT_IDS, Action, AgentObservation, HandoffTask, JointAction
from .occlusion_geometry import distance, target_at_step, visibility_truth


def _move(position: tuple[float, float, float], action: Action, dt: float) -> tuple[float, float, float]:
    action = action.clipped()
    return (position[0] + action.vx * dt, position[1] + action.vy * dt, max(0.5, position[2] + action.vz * dt))


def _message_dict(message: DeliveredMessage, control_hz: float) -> dict[str, Any]:
    return {
        "sender": message.sender,
        "receiver": message.receiver,
        "payload": message.payload,
        "sent_timestamp_ms": int(round(message.sent_t * 1000.0 / control_hz)),
        "deliver_timestamp_ms": None if message.deliver_t is None else int(round(message.deliver_t * 1000.0 / control_hz)),
        "bytes": message.bytes,
        "dropped": message.dropped,
    }


class HandoffEnv:
    """Dependency-free fixture for geometry, protocol, policy, and evaluator tests."""

    simulator = "mock"

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def __init__(self, task: HandoffTask | None = None, *, seed: int = 0, communication: CommunicationChannel | None = None) -> None:
        self.task = task or HandoffTask()
        self.seed = int(seed)
        configured = communication or CommunicationChannel(control_hz=self.task.control_hz, seed=seed)
        self._communication_config = (configured.bandwidth, configured.latency_steps, configured.drop_probability)
        self.reset(seed)

    def reset(self, seed: int | None = None, task_spec: HandoffTask | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        if task_spec is not None:
            self.task = task_spec
        self.t = 0
        self.positions = {key: tuple(value) for key, value in self.task.initial_positions.items()}
        self.velocities = {key: (0.0, 0.0, 0.0) for key in AGENT_IDS}
        self.yaws = {key: 0.0 for key in AGENT_IDS}
        self.path_lengths = {key: 0.0 for key in AGENT_IDS}
        self.minimum_separation = distance(self.positions["uav_0"], self.positions["uav_1"])
        self.events: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.identity_streak = {key: 0 for key in AGENT_IDS}
        self.visible_history: list[set[str]] = []
        self.last_messages: dict[str, dict[str, Any] | None] = {key: None for key in AGENT_IDS}
        self.pre_position_t: int | None = None
        self.occlusion_start_t: int | None = None
        self.handoff_request_t: int | None = None
        self.handoff_delivery_t: int | None = None
        self.recovery_t: int | None = None
        self.handoff_t: int | None = None
        self.verify_t: int | None = None
        self.done = False
        bandwidth, latency_steps, drop_probability = self._communication_config
        self.communication = CommunicationChannel(
            bandwidth_bytes_per_second=bandwidth,
            latency_ms=latency_steps * 1000.0 / self.task.control_hz,
            drop_probability=drop_probability,
            control_hz=self.task.control_hz,
            seed=self.seed,
        )
        return self._observation()

    def _target(self) -> tuple[float, float, float]:
        return target_at_step(self.task.target_start, self.task.target_end, self.t, self.task.horizon)

    def _visibility(self, target: tuple[float, float, float]) -> dict[str, bool]:
        return visibility_truth(
            self.positions,
            target,
            occluder_center=self.task.occluder_center,
            occluder_half_extents=self.task.occluder_half_extents,
            max_range=self.task.visibility_range,
        )

    def _observation(self) -> dict[str, Any]:
        return {
            "task": {
                "task_id": self.task.task_id,
                "instruction": self.task.instruction,
            },
            "timestamp_ms": int(round(self.t * 1000.0 / self.task.control_hz)),
            "observations": {
                agent_id: AgentObservation(
                    agent_id=agent_id,
                    rgb_uri=f"gazebo://camera/{agent_id}/frame_{self.t:06d}.jpg",
                    own_position=self.positions[agent_id],
                    own_velocity=self.velocities[agent_id],
                    own_yaw=self.yaws[agent_id],
                    last_message=self.last_messages[agent_id],
                ).to_policy_dict()
                for agent_id in AGENT_IDS
            },
            "communication_budget": self.communication.bandwidth,
        }

    def _event(self, event_type: str, agent_ids: list[str], *, success: bool, start_t: int | None = None, attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "event_type": event_type,
            "agent_ids": agent_ids,
            "start_t": self.t if start_t is None else int(start_t),
            "end_t": self.t,
            "success": bool(success),
        }
        if attributes:
            event["attributes"] = dict(attributes)
        return event

    def step(self, joint_action: JointAction | Mapping[str, Any], message: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode already terminated; call reset")
        if not isinstance(joint_action, JointAction):
            joint_action = JointAction(
                uav_0=joint_action.get("uav_0", Action()),
                uav_1=joint_action.get("uav_1", Action()),
                message=joint_action.get("message", message),
            )
        joint_action = replace(joint_action, message=joint_action.message if joint_action.message is not None else message).clipped()
        step_events: list[dict[str, Any]] = []
        dt = 1.0 / self.task.control_hz
        previous = dict(self.positions)
        for agent_id, action in (("uav_0", joint_action.uav_0), ("uav_1", joint_action.uav_1)):
            self.positions[agent_id] = _move(self.positions[agent_id], action, dt)
            self.velocities[agent_id] = action.as_vector()[:3]
            self.yaws[agent_id] += action.yaw_rate * dt
            self.path_lengths[agent_id] += distance(previous[agent_id], self.positions[agent_id])
        separation = distance(self.positions["uav_0"], self.positions["uav_1"])
        self.minimum_separation = min(self.minimum_separation, separation)

        sent_messages: list[DeliveredMessage] = []
        if joint_action.message is not None:
            envelope = joint_action.message
            sender = str(envelope.get("sender", "uav_0"))
            receiver = str(envelope.get("receiver", "uav_1"))
            payload = envelope.get("payload", envelope)
            if sender not in AGENT_IDS:
                raise ValueError(f"invalid message sender: {sender}")
            if receiver not in {*AGENT_IDS, "broadcast"}:
                raise ValueError(f"invalid message receiver: {receiver}")
            sent = self.communication.send(sender=sender, receiver=receiver, payload=payload, t=self.t)
            sent_messages.append(sent)
            intent = payload.get("intent") if isinstance(payload, Mapping) else None
            if intent == "handoff_request":
                if self.handoff_request_t is None:
                    self.handoff_request_t = self.t
                step_events.append(self._event("handoff_request", [sender, receiver], success=not sent.dropped, attributes={"dropped": sent.dropped, "bytes": sent.bytes}))
            elif intent == "handoff_ack":
                step_events.append(self._event("handoff_ack_sent", [sender, receiver], success=not sent.dropped, attributes={"dropped": sent.dropped, "bytes": sent.bytes}))

        delivered_messages: list[DeliveredMessage] = []
        for agent_id in AGENT_IDS:
            for delivered in self.communication.receive(receiver=agent_id, t=self.t):
                delivered_messages.append(delivered)
                payload = delivered.payload
                self.last_messages[agent_id] = dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
                intent = payload.get("intent") if isinstance(payload, Mapping) else None
                if intent == "handoff_request" and agent_id == "uav_1":
                    if self.handoff_delivery_t is None:
                        self.handoff_delivery_t = self.t
                    step_events.append(self._event("handoff_delivery", [delivered.sender, agent_id], success=True, start_t=delivered.sent_t, attributes={"latency_steps": self.t - delivered.sent_t}))
                elif intent == "handoff_ack" and agent_id == "uav_0":
                    step_events.append(self._event("handoff_ack_delivery", [delivered.sender, agent_id], success=True, start_t=delivered.sent_t, attributes={"latency_steps": self.t - delivered.sent_t}))

        target = self._target()
        staged = distance(self.positions["uav_1"], self.task.staging_position) <= self.task.staging_tolerance
        if staged and self.pre_position_t is None:
            self.pre_position_t = self.t
            step_events.append(self._event("pre_position", ["uav_1"], success=True))

        visibility = self._visibility(target)
        visible = {agent_id for agent_id, is_visible in visibility.items() if is_visible}
        self.visible_history.append(visible)
        for agent_id in AGENT_IDS:
            self.identity_streak[agent_id] = self.identity_streak[agent_id] + 1 if visibility[agent_id] else 0

        if not visibility["uav_0"] and visibility["uav_1"] and self.occlusion_start_t is None:
            self.occlusion_start_t = self.t
            step_events.append(self._event("occlusion_start", ["uav_0"], success=True, attributes={"target_identity": self.task.target_identity}))
        if self.occlusion_start_t is not None and self.recovery_t is None and self.identity_streak["uav_1"] >= self.task.required_identity_frames:
            self.recovery_t = self.t
            step_events.append(self._event("target_recovery", ["uav_1"], success=True, start_t=self.occlusion_start_t, attributes={"recovery_latency_steps": self.t - self.occlusion_start_t, "target_identity": self.task.target_identity}))
        if self.handoff_t is None and self.handoff_delivery_t is not None and self.recovery_t is not None:
            self.handoff_t = self.t
            start_t = self.occlusion_start_t if self.occlusion_start_t is not None else self.handoff_delivery_t
            step_events.append(self._event("handoff", list(AGENT_IDS), success=True, start_t=start_t, attributes={"latency_steps": self.t - start_t, "target_identity": self.task.target_identity}))

        passed_occluder = target[0] > self.task.occluder_center[0] + self.task.occluder_half_extents[0]
        if self.verify_t is None and self.handoff_t is not None and passed_occluder and self.identity_streak["uav_1"] >= self.task.required_identity_frames:
            self.verify_t = self.t
            step_events.append(self._event("verify", ["uav_1"], success=True, start_t=self.handoff_t, attributes={"target_identity": self.task.target_identity}))

        collision = separation < self.task.collision_distance
        if collision:
            step_events.append(self._event("collision", list(AGENT_IDS), success=False, attributes={"separation_m": separation}))
        self.events.extend(step_events)
        self.t += 1
        interruption = self._longest_interruption()
        success = self.verify_t is not None and interruption <= self.task.interruption_limit_steps and not collision
        self.done = collision or success or self.t >= self.task.horizon
        if self.t >= self.task.horizon and not success and not collision:
            timeout_event = {
                "event_type": "timeout",
                "agent_ids": list(AGENT_IDS),
                "start_t": self.t - 1,
                "end_t": self.t - 1,
                "success": False,
            }
            step_events.append(timeout_event)
            self.events.append(timeout_event)
        termination_reason = "collision" if collision else "success" if success else "timeout" if self.done else "running"
        truth = {
            "target_pose": list(target),
            "target_visible_to": sorted(visible),
            "target_visibility": visibility,
            "target_identity": self.task.target_identity,
            "occluded_for_uav_0": not visibility["uav_0"],
            "collision": collision,
            "minimum_separation_m": self.minimum_separation,
        }
        info = {
            "success": success,
            "collision": collision,
            "termination_reason": termination_reason,
            "visible_agents": sorted(visible),
            "longest_interruption_steps": interruption,
            "events": step_events,
            "communication": {"bytes_sent": self.communication.bytes_sent, "dropped": self.communication.dropped},
            "truth": truth,
        }
        observation = self._observation()
        self.steps.append({
            "t": self.t - 1,
            "observation": observation,
            "action": joint_action,
            "info": info,
            "sent_messages": sent_messages,
            "delivered_messages": delivered_messages,
        })
        return observation, (1.0 if success else -1.0 if collision else 0.0), self.done, info

    def _longest_interruption(self) -> int:
        longest = current = 0
        for visible in self.visible_history:
            if visible:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    def _metrics(self) -> dict[str, Any]:
        recovery_latency = self.task.horizon
        if self.occlusion_start_t is not None and self.recovery_t is not None:
            recovery_latency = self.recovery_t - self.occlusion_start_t
        handoff_latency = self.task.horizon
        if self.occlusion_start_t is not None and self.handoff_t is not None:
            handoff_latency = self.handoff_t - self.occlusion_start_t
        collision = any(event["event_type"] == "collision" for event in self.events)
        return {
            "task_success": float(self.verify_t is not None and not collision and self._longest_interruption() <= self.task.interruption_limit_steps),
            "pre_position_success": float(self.pre_position_t is not None),
            "handoff_success": float(self.handoff_t is not None),
            "confirmation_success": float(self.verify_t is not None),
            "recovery_latency_steps": float(recovery_latency),
            "handoff_latency_steps": float(handoff_latency),
            "longest_interruption_steps": float(self._longest_interruption()),
            "communication_bytes": float(self.communication.bytes_sent),
            "communication_dropped": float(self.communication.dropped),
            "collision_rate": float(collision),
            "minimum_separation_m": float(self.minimum_separation),
            "path_length_uav_0": float(self.path_lengths["uav_0"]),
            "path_length_uav_1": float(self.path_lengths["uav_1"]),
            "path_length": float(sum(self.path_lengths.values())),
            "flight_time_s": float(len(self.steps) / self.task.control_hz),
        }

    def episode_record(self) -> dict[str, Any]:
        last = self.steps[-1]["info"] if self.steps else {"success": False, "termination_reason": "simulator_error", "truth": {}}
        reset_hash = hashlib.sha256(repr(self.task.to_dict()).encode("utf-8") + str(self.seed).encode("ascii")).hexdigest()
        steps = []
        for item in self.steps:
            serialized_observations = {}
            for agent_id in AGENT_IDS:
                policy_observation = item["observation"]["observations"][agent_id]
                own_state = policy_observation["own_state"]
                serialized_observations[agent_id] = {
                    "rgb_uri": policy_observation["rgb_uri"],
                    "own_state": own_state["position"] + own_state["velocity"] + [own_state["yaw"]],
                    "health": own_state["health"],
                    "last_message": policy_observation["last_message"],
                }
            sent = [_message_dict(message, self.task.control_hz) for message in item["sent_messages"]]
            delivered = [_message_dict(message, self.task.control_hz) for message in item["delivered_messages"]]
            steps.append({
                "t": item["t"],
                "timestamp_ms": int(round(item["t"] * 1000.0 / self.task.control_hz)),
                "observations": serialized_observations,
                "actions": {
                    "uav_0": {"control": list(item["action"].uav_0.as_vector()), "valid": True},
                    "uav_1": {"control": list(item["action"].uav_1.as_vector()), "valid": True},
                },
                "communication": {
                    "messages": sent,
                    "sent": sent,
                    "delivered": delivered,
                    "bytes_sent": item["info"]["communication"]["bytes_sent"],
                    "dropped": item["info"]["communication"]["dropped"],
                },
                "events": item["info"]["events"],
                "truth": item["info"]["truth"],
            })
        return {
            "schema_version": "0.2",
            "episode_id": f"{self.task.task_id}__seed_{self.seed}",
            "task_id": self.task.task_id,
            "split": self.task.split,
            "seed": self.seed,
            "agent_ids": list(AGENT_IDS),
            "task_spec": self.task.to_dict(),
            "environment": {
                "simulator": self.simulator,
                "version": "kinematic-fixture-0.3",
                "map_id": "handoff_safe_route_fixture_v2",
                "reset_hash": reset_hash,
                "ros_distro": "none",
                "vehicle_namespaces": list(AGENT_IDS),
                "transport": "deterministic-channel",
                "control_hz": self.task.control_hz,
                "setpoint_hz": 20.0,
            },
            "steps": steps,
            "outcome": {
                "terminated": self.done,
                "termination_reason": last["termination_reason"],
                "success": bool(last["success"]),
                "metrics": self._metrics(),
            },
        }
