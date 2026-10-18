"""Opt-in command-response perturbations, NOT an autonomous teammate policy.

Only UAV1 waypoint requests change. No target, image, task binding, report,
future state or evaluator feedback enters this component. Delays use simulation
time at the controller request boundary, not wall time or camera capture time.
These records prove requested command transformations, NOT physical execution.
"""

from collections import deque
from dataclasses import dataclass, replace

from .protocol import ModelJointAction, WaypointDelta
from .language_tasks import exact_keys, integer

RESPONSE_VERSION = "aerowam-teammate-response/1-draft"
RESPONSE_PROFILES = ("nominal", "cautious", "delayed")


@dataclass(frozen=True)
class ResponseSpec:
    profile: str = "nominal"

    def __post_init__(self):
        if self.profile not in RESPONSE_PROFILES:
            raise ValueError("Unknown teammate response profile")

    @property
    def gain(self):
        return .5 if self.profile == "cautious" else 1.

    @property
    def delay_ns(self):
        return 750_000_000 if self.profile == "delayed" else 0

    def private_metadata(self):
        return {"schema": RESPONSE_VERSION, "profile": self.profile, "agent": "uav_1",
                "waypoint_gain": self.gain, "delay_sim_ns": self.delay_ns,
                "delay_release": "latest_eligible_request_at_next_decision_boundary",
                "scope": "waypoint_request_only_not_messages_reports_or_safety",
                "autonomous_teammate": False, "physical_execution_verified": False}


class TeammateResponse:
    def __init__(self, spec=ResponseSpec()):
        if not isinstance(spec, ResponseSpec):
            raise TypeError("ResponseSpec required")
        self.spec = spec
        self.pending = deque()
        self.last_stamp = -1
        self.step = 0
        self.held = WaypointDelta()
        self.source_step = None

    def issue(self, action, *, sim_stamp_ns):
        if not isinstance(action, ModelJointAction):
            raise TypeError("ModelJointAction required")
        integer(sim_stamp_ns, "controller-boundary simulation stamp")
        if sim_stamp_ns <= self.last_stamp:
            raise ValueError("Controller-boundary simulation time must advance")
        # Snapshot the same clipped request representation used by the protocol.
        requested = action.to_wire()
        peer = WaypointDelta(*requested["waypoints"]["uav_1"])
        step = self.step + 1
        if self.spec.delay_ns:
            if len(self.pending) >= 256:
                raise RuntimeError("Teammate response queue budget exceeded")
            self.pending.append((sim_stamp_ns, step, peer))
            while self.pending and self.pending[0][0] <= sim_stamp_ns - self.spec.delay_ns:
                _, self.source_step, self.held = self.pending.popleft()
            peer = self.held
        else:
            self.source_step = step
        if self.spec.gain != 1.:
            peer = WaypointDelta(*(self.spec.gain * value for value in peer.vector()))
        # Nominal is an exact no-op, including the original action object.
        issued = action if self.spec.profile == "nominal" else replace(action, uav_1=peer)
        row = {"step": step, "sim_stamp_ns": sim_stamp_ns,
               "requested_action": requested, "issued_action": issued.to_wire(),
               "peer_source_step": self.source_step}
        self.last_stamp, self.step = sim_stamp_ns, step
        return issued, row


def request_control(control, transform, action, *, sim_stamp_ns, trace):
    """RUN-only boundary. Landing/safety commands must bypass this function."""
    issued, row = transform.issue(action, sim_stamp_ns=sim_stamp_ns)
    control.request(issued)
    trace.append(row)  # Record only after the controller accepted the request.
    return issued


def audit_response_trace(spec, rows, *, controller_records=None):
    """Independent algebra/causality check; never a flight/task success score."""
    if not isinstance(spec, ResponseSpec):
        raise TypeError("ResponseSpec required")
    previous, history, changed = -1, [], 0
    for step, row in enumerate(rows, 1):
        exact_keys(row, {"step", "sim_stamp_ns", "requested_action", "issued_action", "peer_source_step"})
        integer(row["step"], "step")
        integer(row["sim_stamp_ns"], "simulation stamp")
        if row["step"] != step or row["sim_stamp_ns"] <= previous:
            raise ValueError("Invalid response trace ordering")
        for field in ("requested_action", "issued_action"):
            payload = row[field]
            exact_keys(payload, {"protocol", "waypoints", "intent"})
            exact_keys(payload["waypoints"], {"uav_0", "uav_1"})
            if ModelJointAction.from_wire(payload).to_wire() != payload:
                raise ValueError("Response trace action is not canonical")
        request, issued = row["requested_action"], row["issued_action"]
        if request["waypoints"]["uav_0"] != issued["waypoints"]["uav_0"] or request["intent"] != issued["intent"]:
            raise ValueError("Teammate response changed UAV0 or intent")
        history.append(row)
        eligible = [r for r in history if r["sim_stamp_ns"] <= row["sim_stamp_ns"] - spec.delay_ns]
        source = eligible[-1] if eligible else None
        expected_source = source["step"] if source else None
        if row["peer_source_step"] is not None:
            integer(row["peer_source_step"], "peer source step")
            if row["peer_source_step"] == 0:
                raise ValueError("Invalid peer source step")
        expected = ([spec.gain * v for v in source["requested_action"]["waypoints"]["uav_1"]]
                    if source else [0.] * 4)
        if row["peer_source_step"] != expected_source or issued["waypoints"]["uav_1"] != expected:
            raise ValueError("Incorrect or noncausal teammate response")
        changed += request["waypoints"]["uav_1"] != issued["waypoints"]["uav_1"]
        previous = row["sim_stamp_ns"]
    if controller_records is not None:
        if len(controller_records) != len(history):
            raise ValueError("Missing or extra controller request records")
        for row, (request, issued) in zip(history, controller_records):
            if row["requested_action"] != request or row["issued_action"] != issued:
                raise ValueError("Response trace differs from recorded controller requests")
    return {"schema": RESPONSE_VERSION, "trace_valid": True, "requests": len(history),
            "changed_peer_requests": changed, "nontrivial_perturbation_observed": changed > 0,
            "controller_records_cross_checked": controller_records is not None,
            "physical_execution_verified": False, "task_success_evaluated": False}
