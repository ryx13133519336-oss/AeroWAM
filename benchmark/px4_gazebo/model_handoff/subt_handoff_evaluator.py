"""Private role-free handoff event state machine for a future SubT executor.

Frame evidence must come from the execution harness. In particular,
``fully_occluded`` requires a physical whole-target occlusion certificate;
zero raster support alone is insufficient. This module does not admit flight.
"""

import math

from .subt_handoff_route import AGENTS


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    return intersection / (area(a)+area(b)-intersection) if intersection else 0.


def box(value):
    if value is None:
        return
    if (not isinstance(value, (tuple,list)) or len(value) != 4
            or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value)):
        raise ValueError("Finite normalized box required")
    x1,y1,x2,y2 = value
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError("Invalid box")


class SubTHandoffEvaluator:
    def __init__(self):
        self.source = self.receiver = None
        self.request_ns = self.accept_ns = self.occluded_ns = self.track_ns = None
        self.last_ns = None
        self.complete = False
        self.accepted_events, self.rejected_events = [], []

    def step(self, stamp_ns, views, events):
        if type(stamp_ns) is not int or stamp_ns < 0 or self.last_ns is not None and stamp_ns <= self.last_ns:
            raise ValueError("Strictly increasing frame stamps required")
        if set(views) != set(AGENTS):
            raise ValueError("Both private views required")
        for view in views.values():
            if set(view) != {"target_bbox_xyxy", "fully_occluded"} or type(view["fully_occluded"]) is not bool:
                raise ValueError("Private view evidence fields invalid")
            box(view["target_bbox_xyxy"])
            if view["fully_occluded"] and view["target_bbox_xyxy"] is not None:
                raise ValueError("Physical full occlusion contradicts visible target")
        # Validate all model claims before state mutation; same-frame event
        # ordering must never create instantaneous discovery/request/confirm.
        seen = set()
        for event in events:
            if (set(event) != {"agent_id", "event", "image_timestamp_ns", "bbox_xyxy"}
                    or event["agent_id"] not in AGENTS or event["agent_id"] in seen
                    or event["image_timestamp_ns"] != stamp_ns
                    or event["event"] not in {"handoff.found", "handoff.handoff_request", "handoff.handoff_accept", "handoff.confirm"}):
                raise ValueError("Model event inventory or stamp invalid")
            seen.add(event["agent_id"])
            box(event["bbox_xyxy"])
        gap = self.last_ns is not None and stamp_ns-self.last_ns > 1_000_000_000
        self.last_ns = stamp_ns
        if self.source:
            if not views[self.source]["fully_occluded"]:
                self.occluded_ns = None
            elif gap or self.occluded_ns is None:
                self.occluded_ns = stamp_ns
        old_phase = (self.source, self.request_ns, self.accept_ns)
        grounded = {e["agent_id"] for e in events if iou(e["bbox_xyxy"], views[e["agent_id"]]["target_bbox_xyxy"]) >= .5}
        if self.receiver and self.accept_ns is not None:
            if gap or self.receiver not in grounded or self.occluded_ns is None:
                self.track_ns = None
            elif self.track_ns is None:
                self.track_ns = stamp_ns
        found = [e for e in events if e["event"] == "handoff.found" and e["agent_id"] in grounded]
        for event in events:
            actor, kind = event["agent_id"], event["event"]
            valid = False
            if not self.complete:
                if kind == "handoff.found" and old_phase[0] is None and len(found) == 1 and actor in grounded:
                    self.source = actor
                    self.receiver = next(a for a in AGENTS if a != actor)
                    valid = True
                elif (kind == "handoff.handoff_request" and old_phase[0] is not None
                      and actor == self.source and old_phase[1] is None):
                    self.request_ns = stamp_ns
                    valid = True
                elif (kind == "handoff.handoff_accept" and old_phase[1] is not None
                      and stamp_ns >= self.request_ns+500_000_000 and old_phase[2] is None
                      and actor == self.receiver and actor in grounded):
                    self.accept_ns = stamp_ns
                    valid = True
                elif (kind == "handoff.confirm" and old_phase[2] is not None
                      and actor == self.receiver and actor in grounded
                      and stamp_ns >= self.accept_ns+500_000_000 and self.occluded_ns is not None
                      and self.track_ns is not None and stamp_ns-self.track_ns >= 1_000_000_000
                      and stamp_ns-self.occluded_ns >= 1_000_000_000):
                    self.complete = valid = True
            (self.accepted_events if valid else self.rejected_events).append({
                "stamp_ns": stamp_ns, "agent_id": actor, "event": kind})

    def result(self):
        return {"schema": "aerowam-subt-handoff-private-evaluator/1", "success": self.complete,
                "source_agent": self.source, "receiver_agent": self.receiver,
                "accepted_events": self.accepted_events, "rejected_events": self.rejected_events,
                "flight_admitted": False,
                "scope": "private_harness_evidence_required_no_flight_or_transfer_admission"}
