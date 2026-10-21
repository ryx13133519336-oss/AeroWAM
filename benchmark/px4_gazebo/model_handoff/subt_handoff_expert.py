"""Privileged kinematic demonstration plan, not PX4 control or model results."""

import math

import numpy as np

from .subt_handoff_route import AGENTS, local_triangles, ray_visible, sample
from .subt_urban_geometry import triangles_touch_box


def look_at(eye, point):
    delta = np.asarray(point)-np.asarray(eye)
    if np.linalg.norm(delta) < .01:
        raise ValueError("Look-at target coincides with camera")
    return [*map(float, eye), 0., -math.atan2(delta[2], math.hypot(*delta[:2])), math.atan2(delta[1], delta[0])]


def planned_state(route, elapsed, *, speed=.3, swapped=False, teammate="normal"):
    if (not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(speed) or not 0 < speed <= .4
            or teammate not in {"normal", "delayed"}):
        raise ValueError("Finite elapsed, speed <=.4 and declared teammate behavior required")
    length = route["distance_m"][-1]
    progress = min(length, max(0., elapsed-4.)*speed)
    target = sample(route, progress)
    # Entry observer follows the first flight but cannot see through the upper
    # floor. Exit observer descends its own flight then covers the upper exit.
    first = sample(route, min(13., max(0., progress-3.)))+[0,0,1.2]
    delay = 4. if teammate == "delayed" else 0.
    arrival = max(0., progress-delay)
    exit_distance = length - min(10., arrival*10./14.) if arrival < 14 else min(length, max(length-10., arrival+10.))
    second = sample(route, exit_distance)+[0,0,1.2]
    eyes = [first, second][::-1] if swapped else [first, second]
    return {"elapsed_s": elapsed, "progress_m": progress, "target_enu": target.tolist(),
            "camera_poses_enu": {a: look_at(eye, target) for a,eye in zip(AGENTS, eyes)}}


def duration(route, speed=.3):
    if not math.isfinite(speed) or not 0 < speed <= .4:
        raise ValueError("Finite positive speed <=.4 required")
    return 8. + route["distance_m"][-1]/speed


def minimum_segment_separation(a, b):
    """Minimum length of linearly changing relative position in one segment."""
    a, b = np.asarray(a), np.asarray(b)
    delta = b-a
    alpha = float(np.clip(-np.dot(a,delta)/np.dot(delta,delta), 0., 1.)) if np.dot(delta,delta)>0 else 0.
    return float(np.linalg.norm(a+alpha*delta))


def audit_expert(triangles, route, *, speed=.3, swapped=False, teammate="normal"):
    count = math.ceil(duration(route,speed)/.25)
    states = [planned_state(route,t,speed=speed,swapped=swapped,teammate=teammate)
              for t in np.linspace(0.,duration(route,speed),count+1)]
    extents = np.array([state["camera_poses_enu"][agent][:3] for state in states for agent in AGENTS]
                       + [state["target_enu"] for state in states])
    triangles = local_triangles(triangles, extents.min(0)-1., extents.max(0)+1.)
    issues, visibility = [], []
    min_separation = float("inf")
    # Axis-aligned inflated camera-proxy volumes. Not an x500 rotor/body proof.
    half = np.array([.4,.4,.25])
    for index,(previous,current) in enumerate(zip(states,states[1:])):
        for agent in AGENTS:
            a, b = (np.asarray(s["camera_poses_enu"][agent][:3]) for s in (previous,current))
            if triangles_touch_box(triangles,np.minimum(a,b)-half,np.maximum(a,b)+half):
                issues.append({"code":"camera_proxy_mesh_collision","agent":agent,"time_s":current["elapsed_s"]})
        pairs = [("uav_0","uav_1"),("uav_0",None),("uav_1",None)]
        for first,second in pairs:
            relative = []
            for state in (previous,current):
                p = np.asarray(state["camera_poses_enu"][first][:3])
                q = state["target_enu"] if second is None else state["camera_poses_enu"][second][:3]
                relative.append(p-np.asarray(q))
            separation = minimum_segment_separation(*relative)
            min_separation = min(min_separation,separation)
            if separation < 1.:
                issues.append({"code":"proxy_separation","time_s":current["elapsed_s"]})
        visibility.append({"elapsed_s":current["elapsed_s"], "visible":{a:ray_visible(
            triangles,current["camera_poses_enu"][a][:3],current["target_enu"]) for a in AGENTS}})
    leader = "uav_1" if swapped else "uav_0"
    receiver = next(a for a in AGENTS if a!=leader)
    expected = [(True,False),(True,True),(False,True)]
    stages, previous, start, last = [], None, None, None
    for row in visibility:
        pattern = tuple(row["visible"][a] for a in (leader,receiver))
        if pattern != previous:
            if previous is not None:
                stages.append({"pattern":list(previous),"duration_s":last-start,"start_s":start})
            previous,start = pattern,row["elapsed_s"]
        last=row["elapsed_s"]
    if previous is not None:
        stages.append({"pattern":list(previous),"duration_s":last-start,"start_s":start})
    ordered, selected = 0, []
    for stage in stages:
        if ordered<3 and tuple(stage["pattern"])==expected[ordered] and stage["duration_s"]>=1.:
            selected.append(stage)
            ordered+=1
    if ordered!=3:
        issues.append({"code":"ordered_ray_visibility_stages_missing"})
    return {"passed":not issues,"issues":issues,"ray_stages":stages,"selected_stages":selected,
            "minimum_proxy_separation_m":min_separation,"proxy_half_extent_m":half.tolist(),
            "scope":"offline_proxy_volume_and_center_ray_only_rendering_required",
            "flight_admitted":False,"expert_is_privileged":True,"policy_evaluated":False}


class EventExpert:
    """Causal oracle labels from current PRIVATE rendered visibility, not a detector.

    No future trajectory is used here. Public inputs contain RGB/proprioception
    and previously delivered claims only. Current oracle labels stay supervision.
    """

    def __init__(self, delay_s=.5):
        if not math.isfinite(delay_s) or delay_s<0:
            raise ValueError("Nonnegative finite message delay required")
        self.delay_s=delay_s
        self.owner=None
        self.receiver=None
        self.pending=False
        self.accepted=False
        self.complete=False
        self.request_stamp=None
        self.accept_stamp=None
        self.last_stamp=-1.
        self.seen_since={a:None for a in AGENTS}
        self.hidden_since={a:None for a in AGENTS}

    def step(self, elapsed, visible):
        if not math.isfinite(elapsed) or elapsed<=self.last_stamp or set(visible)!=set(AGENTS):
            raise ValueError("Increasing timestamps and both visibility inputs required")
        self.last_stamp=elapsed
        for agent in AGENTS:
            if visible[agent]:
                if self.seen_since[agent] is None:self.seen_since[agent]=elapsed
                self.hidden_since[agent]=None
            else:
                if self.hidden_since[agent] is None:self.hidden_since[agent]=elapsed
                self.seen_since[agent]=None
        events=[]
        def stable(agent):
            return self.seen_since[agent] is not None and elapsed-self.seen_since[agent]>=1.
        if self.owner is None:
            # Only unambiguous initial acquisition assigns a temporary owner.
            candidates=[a for a in AGENTS if stable(a)]
            if len(candidates)==1:
                self.owner=candidates[0]
                self.receiver=next(a for a in AGENTS if a!=self.owner)
                events.append({"agent_id":self.owner,"event":"found"})
        elif not self.complete:
            lost=self.hidden_since[self.owner] is not None and elapsed-self.hidden_since[self.owner]>=1.
            if not self.pending and (stable(self.receiver) or lost):
                self.pending=True
                self.request_stamp=elapsed
                events.append({"agent_id":self.owner,"event":"handoff_request"})
            elif self.pending and not self.accepted and elapsed>=self.request_stamp+self.delay_s and stable(self.receiver):
                self.accepted=True
                self.accept_stamp=elapsed
                events.append({"agent_id":self.receiver,"event":"handoff_accept"})
            elif self.accepted and elapsed>self.accept_stamp and lost and stable(self.receiver):
                self.complete=True
                events.append({"agent_id":self.receiver,"event":"confirm"})
        return events
