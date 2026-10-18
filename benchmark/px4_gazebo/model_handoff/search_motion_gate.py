"""Private additional acceptance for opt-in moving search; never policy input.

Range is a spawn-calibrated receive-time odometry estimate, not exact camera
exposure truth. The original language evaluator still validates every box,
identity, freshness and event. No original rgb_scan acceptance is changed.
"""

import math

from ..ros2_px4_adapter import ned_displacement_to_initial_body_flu
from .protocol import AGENT_IDS
from .search_pilot_scene import SPAWNS_NED

MOTION_RULE_VERSION = "search_approach_acceptance/1"


def private_local_goals(tree, target_id, initial_yaws):
    scene = tree.getroot().find("world")
    x, y, z, *_ = map(float, scene.findtext(f"model[@name='{target_id}']/pose").split())
    return {agent: ned_displacement_to_initial_body_flu(
        (y - SPAWNS_NED[i][0], x - SPAWNS_NED[i][1], -z), initial_yaws[i])
        for i, agent in enumerate(AGENT_IDS)}


class MotionSearchGate:
    def __init__(self, goals):
        self.goals = goals
        self.initial = None
        self.receiver_start = None
        self.found_stamp = None
        self.span = None
        self.metrics = {}

    def check(self, observation, frames, reports, target_id, now):
        positions = {a: observation.states[a].position[:2] for a in AGENT_IDS}
        if self.initial is None:
            self.initial = dict(positions)
        receiver_report = next((r for r in reports if r.agent_id == "uav_1"), None)
        if receiver_report is None:
            self.span = None
        for report in reports:
            a, p = report.agent_id, positions[report.agent_id]
            if report.event == "found" and a == "uav_0" and self.found_stamp is None:
                moved = math.dist(p, self.initial[a])
                closer = math.dist(self.initial[a], self.goals[a][:2]) - math.dist(p, self.goals[a][:2])
                self.metrics.update(scout_displacement_m=moved, scout_range_reduction_m=closer)
                if moved < .8 or closer < .7:
                    return "found_without_actual_scout_approach"
                self.receiver_start, self.found_stamp = positions["uav_1"], now
                self.span = None
            if a != "uav_1" or self.found_stamp is None:
                continue
            moved = math.dist(p, self.receiver_start)
            closer = math.dist(self.receiver_start, self.goals[a][:2]) - math.dist(p, self.goals[a][:2])
            box = frames[a].objects[target_id].visible_bbox
            stamp = report.image_timestamp_ns
            ready = (moved >= .8 and closer >= .7 and stamp > self.found_stamp and box is not None
                     and abs((box[0] + box[2]) / 2 - .5) <= .14
                     and math.hypot(*observation.states[a].velocity[:2]) <= .22)
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
            if report.event == "confirm" and (not ready or not self.span or self.span[2] < 3 or dwell < 1.):
                return "confirm_without_actual_receiver_approach_and_dwell"
        return None
