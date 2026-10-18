"""Opt-in open-arena RGB approach demo, NOT a learned navigation policy.

No scene imports, shared-world coordinates, private labels or expert routes.
Each aircraft independently centers a public color box and moves forward.
The centralized controller shares only its own emitted found-event state.
Stationary objects, one object per color, no obstacle avoidance are explicit
restrictions. A sweep can see the target; translation is a task requirement,
not evidence that occlusion-aware exploration has been solved.
"""

import math

from .grounded_interface import GroundedAction
from .language_tasks import TargetReport
from .protocol import AGENT_IDS, ModelJointAction, WaypointDelta
from .search_reporter import color_box, requested_color, reportable_box

MOTION_POLICY_VERSION = "rgb_search_approach/2"


def clip(value, limit):
    return max(-limit, min(limit, value))


def hold(state, anchor, yaw=0.):
    # Public yaw is NED-clockwise relative to initial yaw. Position axes are
    # initial-body FLU; action axes are CURRENT-body FLU (positive yaw left).
    ex, ey = (anchor[k] - state.position[k] for k in (0, 1))
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return WaypointDelta(clip(.4 * (c * ex - s * ey), .15),
                         clip(.4 * (s * ex + c * ey), .15),
                         clip(.4 * (2. - state.position[2]), .12), yaw)


class RGBSearchPolicy:
    def __init__(self):
        self.episode = None
        self.diagnostics = {}

    def __call__(self, observation):
        color = requested_color(observation.instruction)
        if self.episode != observation.episode_id:
            self.episode, self.instruction = observation.episode_id, observation.instruction
            self.initial = {a: tuple(observation.states[a].position) for a in AGENT_IDS}
            self.anchors, self.widths, self.stable, self.held = {}, {}, {}, {}
            self.settling = set()
            self.previous = {a: -1 for a in AGENT_IDS}
            self.found = self.confirmed = False
        if self.instruction != observation.instruction:
            raise ValueError("Instruction changed within one episode")
        found_before = self.found
        actions, reports, diagnostics = {}, [], {}
        for agent in AGENT_IDS:
            state, stamp = observation.states[agent], observation.image_timestamp_ns[agent]
            if stamp <= self.previous[agent]:
                raise ValueError("RGB search requires advancing camera frames")
            self.previous[agent] = stamp
            displacement = math.dist(state.position[:2], self.initial[agent][:2])
            if displacement > 2.5:
                raise RuntimeError("RGB search local travel budget exceeded; land, do not guess a route")
            box = color_box(observation.rgb_jpeg[agent], color)
            report_ready = reportable_box(box)
            action = hold(state, self.held.get(agent, self.initial[agent]))
            phase, progress = "WAIT_SCOUT", 0.
            active = agent == "uav_0" or found_before
            finished = self.found if agent == "uav_0" else self.confirmed
            event = "observe"
            if finished:
                phase = "HOLD_FOUND" if agent == "uav_0" else "CONFIRMED"
                if box:
                    action = hold(state, self.held[agent], clip(-.5 * ((box[0] + box[2]) / 2 - .5), .10))
            elif active:
                if box is None:
                    phase, action = "SCAN", hold(state, self.held.get(agent, self.initial[agent]), -.12)
                    self.stable.pop(agent, None)
                else:
                    center = (box[0] + box[2]) / 2 - .5
                    width = box[2] - box[0]
                    yaw = clip(-.7 * center, .12)
                    phase, action = "CENTER", hold(state, self.held.get(agent, self.initial[agent]), yaw)
                    if report_ready:
                        if agent not in self.anchors:
                            self.anchors[agent] = tuple(state.position)
                            self.widths[agent] = width
                        progress = math.dist(state.position[:2], self.anchors[agent][:2])
                        reached = progress >= 1.1 and width >= 1.12 * self.widths[agent]
                        if reached or agent in self.settling:
                            if agent not in self.settling:
                                self.held[agent] = tuple(state.position)
                                self.settling.add(agent)
                            action, phase = hold(state, self.held[agent], yaw), "STABILIZE"
                            speed = math.hypot(*state.velocity[:2])
                            if speed <= .18:
                                start, end, count = self.stable.get(agent, (stamp, stamp, 0))
                                if stamp - end > 800_000_000:
                                    start, count = stamp, 0
                                self.stable[agent] = start, stamp, count + 1
                                if count + 1 >= 3 and stamp - start >= 1_200_000_000:
                                    if agent == "uav_0":
                                        event, self.found = "found", True
                                    else:
                                        event, self.confirmed = "confirm", True
                            else:
                                self.stable.pop(agent, None)
                        elif width < .28:
                            phase = "APPROACH"
                            # If the next image loses the box, stop near the
                            # latest measured position rather than fly home.
                            self.held[agent] = tuple(state.position)
                            action = WaypointDelta(.22, 0., action.dz, yaw)
                            self.stable.pop(agent, None)
                        else:
                            raise RuntimeError("Object too large before verified approach; refuse further forward motion")
                    else:
                        self.stable.pop(agent, None)
            if report_ready and active:
                reports.append(TargetReport(agent, stamp, box, 1., event))
            actions[agent] = action
            if agent in self.anchors:
                progress = math.dist(state.position[:2], self.anchors[agent][:2])
            span = self.stable.get(agent)
            diagnostics[agent] = {"phase": phase, "xy_m": [round(v, 2) for v in state.position[:2]],
                                  "approach_m": round(progress, 2), "visible": box is not None,
                                  "report_ready": report_ready,
                                  "speed_xy_m_s": round(math.hypot(*state.velocity[:2]), 3),
                                  "stable_dwell_s": (span[1] - span[0]) / 1e9 if span else 0.,
                                  "stable_frames": span[2] if span else 0,
                                  "requested_delta": action.vector()}
        self.diagnostics = diagnostics
        return GroundedAction(ModelJointAction(**actions), tuple(reports))
