"""Opt-in role-free search wiring, not learned role selection or navigation.

The auxiliary reporter uses only public pixels/local state. Its discovery
source is evidence-dependent, never prescribed by agent ID. Private range
acceptance remains separate. Neither component generates control actions.
"""

import math

from .language_tasks import TargetReport
from .protocol import AGENT_IDS
from .search_motion_gate import MotionSearchGate
from .search_reporter import check_search_travel, color_box, reportable_box, requested_color

ROLE_FREE_REPORTER_VERSION = "pilot_rgb_unassigned_approach_reporter/1"
ROLE_FREE_MOTION_VERSION = "search_unassigned_approach_acceptance/1"


class RoleFreeApproachReporter:
    def __init__(self):
        self.episode = None
        self.diagnostics = {}

    def reports(self, observation):
        color = requested_color(observation.instruction)
        if self.episode != observation.episode_id:
            self.episode, self.instruction = observation.episode_id, observation.instruction
            self.initial = {a: tuple(observation.states[a].position) for a in AGENT_IDS}
            self.baselines = dict(self.initial)
            self.anchors, self.widths, self.stable = {}, {}, {}
            self.previous = {a: -1 for a in AGENT_IDS}
            self.source = self.receiver = self.found_stamp = None
            self.confirmed = False
        if observation.instruction != self.instruction:
            raise ValueError("Instruction changed within one episode")
        for agent in AGENT_IDS:
            if observation.image_timestamp_ns[agent] <= self.previous[agent]:
                raise ValueError("Approach reporter requires advancing camera frames")
        check_search_travel(observation, self.initial)
        self.previous = dict(observation.image_timestamp_ns)
        boxes = {a: color_box(observation.rgb_jpeg[a], color) for a in AGENT_IDS}
        eligible, diagnostics = [], {}
        # Compute both agents before selecting an event; list iteration order
        # cannot give UAV0 priority. A simultaneous eligibility tie defers found.
        for agent in AGENT_IDS:
            state, stamp, box = observation.states[agent], observation.image_timestamp_ns[agent], boxes[agent]
            ready = reportable_box(box)
            finished = agent == self.source or self.confirmed
            phase, progress = "EVIDENCE_COMPLETE" if finished else "WAIT_VISIBLE", 0.
            if not finished:
                if not ready:
                    self.stable.pop(agent, None)
                else:
                    if agent not in self.anchors:
                        self.anchors[agent] = tuple(state.position)
                        self.widths[agent] = box[2] - box[0]
                    progress = math.dist(state.position[:2], self.anchors[agent][:2])
                    moved = math.dist(state.position[:2], self.baselines[agent][:2])
                    approached = (progress >= 1.1 and moved >= 1.1
                                  and box[2] - box[0] >= 1.12 * self.widths[agent])
                    phase = "WAIT_APPROACH"
                    if approached and math.hypot(*state.velocity[:2]) <= .18:
                        phase = "STABILIZE"
                        start, end, count = self.stable.get(agent, (stamp, stamp, 0))
                        if stamp - end > 800_000_000:
                            start, count = stamp, 0
                        self.stable[agent] = start, stamp, count + 1
                        if count + 1 >= 3 and stamp - start >= 1_200_000_000:
                            eligible.append(agent)
                    else:
                        self.stable.pop(agent, None)
            span = self.stable.get(agent)
            diagnostics[agent] = {"phase": phase, "visible": box is not None, "report_ready": ready,
                                  "approach_m": round(progress, 3),
                                  "stable_dwell_s": (span[1] - span[0]) / 1e9 if span else 0.,
                                  "stable_frames": span[2] if span else 0}
        events = {}
        if self.source is None and len(eligible) == 1:
            self.source = eligible[0]
            self.receiver = next(a for a in AGENT_IDS if a != self.source)
            self.found_stamp = observation.image_timestamp_ns[self.source]
            events[self.source] = "found"
            # No pre-discovery motion, image growth or dwell counts for peer
            # confirmation; the next fresh frame establishes its image anchor.
            self.baselines[self.receiver] = tuple(observation.states[self.receiver].position)
            for mapping in (self.anchors, self.widths, self.stable):
                mapping.pop(self.receiver, None)
            diagnostics[self.receiver].update(phase="POST_FOUND_RESET", approach_m=0.,
                                               stable_dwell_s=0., stable_frames=0)
        elif self.source is None and len(eligible) > 1:
            for agent in eligible:
                diagnostics[agent]["phase"] = "SIMULTANEOUS_EVIDENCE_DEFERRED"
        elif (not self.confirmed and self.receiver in eligible
              and observation.image_timestamp_ns[self.receiver] > self.found_stamp):
            events[self.receiver], self.confirmed = "confirm", True
        self.diagnostics = diagnostics
        return tuple(TargetReport(a, observation.image_timestamp_ns[a], boxes[a], 1., events.get(a, "observe"))
                     for a in AGENT_IDS if reportable_box(boxes[a]))


class RoleFreeMotionSearchGate(MotionSearchGate):
    role_mode = "unassigned"
    rule_version = ROLE_FREE_MOTION_VERSION

    def __init__(self, goals):
        super().__init__(goals)
        self.source = self.receiver = None

    def check(self, observation, frames, reports, target_id, now):
        positions = {a: observation.states[a].position[:2] for a in AGENT_IDS}
        if self.initial is None:
            self.initial = dict(positions)
        found = [r for r in reports if r.event == "found"]
        if len(found) > 1:
            return "multiple_found_in_one_response"
        if found:
            if self.source is not None:
                return "invalid_search_event_order"
            agent = found[0].agent_id
            moved = math.dist(positions[agent], self.initial[agent])
            closer = (math.dist(self.initial[agent], self.goals[agent][:2])
                      - math.dist(positions[agent], self.goals[agent][:2]))
            self.metrics.update(source_displacement_m=moved, source_range_reduction_m=closer)
            if moved < .8 or closer < .7:
                return "found_without_actual_source_approach"
            self.source = agent
            self.receiver = next(a for a in AGENT_IDS if a != agent)
            self.receiver_start, self.found_stamp = positions[self.receiver], now
            self.span = None
        if any(r.event == "confirm" and (self.source is None or r.agent_id != self.receiver) for r in reports):
            return "invalid_search_event_order"
        peer = next((r for r in reports if r.agent_id == self.receiver), None)
        if peer is None:
            self.span = None
            return None
        agent, p, stamp = self.receiver, positions[self.receiver], peer.image_timestamp_ns
        moved = math.dist(p, self.receiver_start)
        closer = math.dist(self.receiver_start, self.goals[agent][:2]) - math.dist(p, self.goals[agent][:2])
        box = frames[agent].objects[target_id].visible_bbox
        ready = (moved >= .8 and closer >= .7 and stamp > self.found_stamp and box is not None
                 and abs((box[0] + box[2]) / 2 - .5) <= .14
                 and math.hypot(*observation.states[agent].velocity[:2]) <= .22)
        if ready:
            start, end, count = self.span or (stamp, stamp, 0)
            if stamp - end > 800_000_000:
                start, count = stamp, 0
            self.span = start, stamp, count + 1
        else:
            self.span = None
        dwell = (self.span[1] - self.span[0]) / 1e9 if self.span else 0.
        self.metrics.update(receiver_displacement_after_found_m=moved,
                            receiver_range_reduction_m=closer, receiver_stable_dwell_s=dwell)
        if peer.event == "confirm" and (not ready or not self.span or self.span[2] < 3 or dwell < 1.):
            return "confirm_without_actual_receiver_approach_and_dwell"
        return None
