"""Optional public-only movement evidence. Generates NO navigation actions.

This auxiliary reporter is not a learned grounding/event head. Public local
odometry and RGB width growth are only conservative reporting prerequisites;
the private MotionSearchGate still checks actual target-range reduction.
"""

import math

from .language_tasks import TargetReport
from .protocol import AGENT_IDS
from .search_reporter import check_search_travel, color_box, reportable_box, requested_color

APPROACH_REPORTER_VERSION = "pilot_rgb_approach_reporter/1"


class ApproachReporter:
    def __init__(self):
        self.episode = None
        self.diagnostics = {}

    def reports(self, observation):
        color = requested_color(observation.instruction)
        if self.episode != observation.episode_id:
            self.episode, self.instruction = observation.episode_id, observation.instruction
            self.initial = {a: tuple(observation.states[a].position) for a in AGENT_IDS}
            self.baselines = {"uav_0": self.initial["uav_0"]}
            self.anchors, self.widths, self.stable = {}, {}, {}
            self.previous = {a: -1 for a in AGENT_IDS}
            self.found_stamp = None
            self.confirmed = False
        if observation.instruction != self.instruction:
            raise ValueError("Instruction changed within one episode")
        # Validate BOTH cameras before mutating event/evidence history.
        for agent in AGENT_IDS:
            if observation.image_timestamp_ns[agent] <= self.previous[agent]:
                raise ValueError("Approach reporter requires advancing camera frames")
        check_search_travel(observation, self.initial)
        self.previous = dict(observation.image_timestamp_ns)
        found_before = self.found_stamp
        reports, diagnostics = [], {}
        for agent in AGENT_IDS:
            state, stamp = observation.states[agent], observation.image_timestamp_ns[agent]
            box = color_box(observation.rgb_jpeg[agent], color)
            ready = reportable_box(box)
            active = agent == "uav_0" or found_before is not None
            finished = self.found_stamp is not None if agent == "uav_0" else self.confirmed
            event, progress, phase = "observe", 0., "WAIT_SCOUT"
            if active:
                phase = "EVIDENCE_COMPLETE" if finished else "WAIT_VISIBLE"
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
                        # Unlike navigation, reporting cannot brake the model
                        # or latch a reached position while it flies away.
                        if approached and math.hypot(*state.velocity[:2]) <= .18:
                            phase = "STABILIZE"
                            start, end, count = self.stable.get(agent, (stamp, stamp, 0))
                            if stamp - end > 800_000_000:
                                start, count = stamp, 0
                            self.stable[agent] = start, stamp, count + 1
                            if count + 1 >= 3 and stamp - start >= 1_200_000_000:
                                if agent == "uav_0":
                                    event, self.found_stamp = "found", stamp
                                    # Receiver movement BEFORE this own event
                                    # cannot count towards teammate approach.
                                    self.baselines["uav_1"] = tuple(observation.states["uav_1"].position)
                                elif stamp > found_before:
                                    event, self.confirmed = "confirm", True
                        else:
                            self.stable.pop(agent, None)
                if ready:
                    reports.append(TargetReport(agent, stamp, box, 1., event))
            span = self.stable.get(agent)
            diagnostics[agent] = {"phase": phase, "visible": box is not None, "report_ready": ready,
                                  "approach_m": round(progress, 3),
                                  "stable_dwell_s": (span[1] - span[0]) / 1e9 if span else 0.,
                                  "stable_frames": span[2] if span else 0}
        self.diagnostics = diagnostics
        return tuple(reports)
