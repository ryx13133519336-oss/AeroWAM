"""Trusted search episode coordinator; private annotations never reach policy."""

from .grounded_interface import GroundedAction, validate_observed_reports
from .language_evaluator import LanguageEvaluator, ObservedFrame
from .protocol import AGENT_IDS, ModelObservation


class SearchEpisode:
    def __init__(self, bound_task, episode_id, motion_gate=None):
        if bound_task.definition.task != "search" or not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Search task and opaque episode ID required")
        if bound_task.definition.role_mode == "unassigned" and motion_gate is not None:
            raise ValueError("Legacy fixed-role motion gate cannot score an unassigned-role task")
        self.evaluator = LanguageEvaluator(bound_task)
        self.episode_id = episode_id
        self.instruction = bound_task.definition.instruction
        self.previous = {agent: -1 for agent in AGENT_IDS}
        self.actions_requested = 0
        self.motion_gate = motion_gate

    def observation_error(self, reason):
        self.evaluator.invalidate("simulator_error", reason)

    def annotation_error(self, reason):
        self.evaluator.invalidate("evaluator_error", reason)

    def model_error(self, reason):
        self.evaluator.model_error(reason)

    def validate_capture(self, observation, frames):
        if not isinstance(observation, ModelObservation):
            raise ValueError("Public observation required")
        # Validate the same wire schema the server actually receives.
        ModelObservation.from_wire(observation.to_wire())
        if observation.episode_id != self.episode_id or observation.instruction != self.instruction:
            raise ValueError("Episode/instruction mismatch")
        if set(frames) != set(AGENT_IDS):
            raise ValueError("Two private frame annotations required")
        for agent in AGENT_IDS:
            frame = frames[agent]
            if (not isinstance(frame, ObservedFrame) or frame.agent_id != agent
                    or frame.image_timestamp_ns != observation.image_timestamp_ns[agent]
                    or frame.image_timestamp_ns <= self.previous[agent]
                    or set(frame.objects) != set(self.evaluator.task.object_ids)):
                raise ValueError("Capture/annotation identity, stamp or object coverage mismatch")

    def delivered(self, observation, frames, response, *, now_sim_time_ns):
        """Call only after successful inference on this exact observation.

        Returning an action is a request, not an actuation acknowledgement.
        No evaluator target/visibility result is returned to the policy.
        """
        self.validate_capture(observation, frames)
        if not isinstance(response, GroundedAction):
            raise TypeError("Grounded response required")
        validate_observed_reports(response, observation)
        for agent in AGENT_IDS:
            self.evaluator.register_observed_frame(frames[agent])
            self.previous[agent] = observation.image_timestamp_ns[agent]
        motion_error = (self.motion_gate.check(observation, frames, response.reports,
                                              self.evaluator.task.target_id, now_sim_time_ns)
                        if self.motion_gate is not None else None)
        if motion_error:
            self.evaluator.model_error(motion_error)
        else:
            for report in response.reports:
                self.evaluator.report(report, now_sim_time_ns=now_sim_time_ns)
        self.actions_requested += 1
        return response.action

    def finish(self):
        result = {**self.evaluator.finish(), "actions_requested": self.actions_requested}
        if self.motion_gate is not None:
            from .search_motion_gate import MOTION_RULE_VERSION
            result.update(motion_acceptance=MOTION_RULE_VERSION, motion_metrics=dict(self.motion_gate.metrics),
                          motion_calibration={"private_local_goals": self.motion_gate.goals,
                                              "initial_xy": self.motion_gate.initial,
                                              "receiver_xy_at_found": self.motion_gate.receiver_start,
                                              "found_sim_stamp_ns": self.motion_gate.found_stamp})
        return result
