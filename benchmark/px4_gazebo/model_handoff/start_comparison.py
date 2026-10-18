"""Offline disarmed receive-state diagnostics; never certifies candidate starts.

No simulation, model, clock query, alignment fitting, frame search or score write.
PX4 local NED origins are not known to coincide across independent sessions.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from .clock_probe import PROBE_SCHEMA, analyze
from .language_tasks import exact_keys, finite_number, integer
from .protocol import AGENT_IDS
from .startup_clock import session_quality

SPEC_SCHEMA = "aerowam-startup-comparison-spec/1-draft"
REPORT_SCHEMA = "aerowam-startup-comparison/1-draft"
PURPOSE = "exploratory_saved_disarmed_receive_state_diagnostic"
TOLERANCES = {"position_l2_m", "velocity_l2_m_s", "attitude_geodesic_rad",
              "angular_velocity_l2_rad_s", "quaternion_norm_error"}
DISCRETE = ("pose_frame", "velocity_frame", "reset_counter", "quality", "arming_state",
            "nav_state", "failsafe", "pre_flight_checks_pass")
MISSING_STATE = ("physics_internal_state", "controller_and_estimator_internal_state",
                 "rng_state", "pending_control_and_sensor_queues", "task_and_teammate_state")
MISSING_REQUIREMENTS = ("per_vehicle_common_coordinate_reference", "exposure_time_state_alignment",
                        "decision_and_replayed_prefix_identity", *MISSING_STATE)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    result = json.loads(raw, object_pairs_hook=unique)
    canonical(result)  # Reject NaN/Infinity even in otherwise unused metadata.
    return result


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_spec(spec):
    exact_keys(spec, {"schema", "purpose", "tolerances"})
    if spec["schema"] != SPEC_SCHEMA or spec["purpose"] != PURPOSE:
        raise ValueError("Only exploratory saved startup diagnostics are supported")
    exact_keys(spec["tolerances"], TOLERANCES)
    for key, value in spec["tolerances"].items():
        finite_number(value, key)
        if value < 0:
            raise ValueError("Nonnegative explicit tolerances required")
    if spec["tolerances"]["attitude_geodesic_rad"] > math.pi:
        raise ValueError("Attitude tolerance must be <= pi")
    if spec["tolerances"]["quaternion_norm_error"] >= 1:
        raise ValueError("Quaternion norm tolerance must be < 1")


def _vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("Malformed state vector")
    for item in value:
        finite_number(item, "state component")
    return value


def _state(value):
    exact_keys(value, set(AGENT_IDS))
    for state in value.values():
        for key in ("position_ned_m", "velocity_ned_m_s", "angular_velocity_rad_s"):
            _vector(state[key], 3)
        q = _vector(state["quaternion_wxyz"], 4)
        if not 0 < math.hypot(*q) < math.inf:
            raise ValueError("Invalid quaternion norm")
        for key in ("pose_frame", "velocity_frame", "reset_counter", "arming_state", "nav_state",
                    "timestamp_us", "timestamp_sample_us"):
            integer(state[key], key)
        if type(state["quality"]) is not int:
            raise ValueError("Integer odometry quality required")
        for key in ("failsafe", "pre_flight_checks_pass"):
            if type(state[key]) is not bool:
                raise ValueError("Boolean vehicle status required")
        if state["pose_frame"] != 1 or state["velocity_frame"] != 1:
            raise ValueError("Only declared PX4 NED receive states supported")
        if state["arming_state"] != 1 or state["failsafe"]:
            raise ValueError("Only disarmed non-failsafe startup records supported")


def compare_states(left, right, spec):
    """Pure numerical check only; origins/time/state coverage are not inferred.

    Quaternion geodesic distance uses the shorter q/-q path. Norm errors remain
    separate: normalizing for this calculation does not repair bad evidence.
    Agent IDs are paired as recorded, never permuted to improve agreement.
    """
    left, right, spec = _decode(canonical([left, right, spec]))
    validate_spec(spec)
    _state(left)
    _state(right)
    result = {}
    for agent in AGENT_IDS:
        a, b = left[agent], right[agent]
        qa, qb = a["quaternion_wxyz"], b["quaternion_wxyz"]
        na, nb = math.hypot(*qa), math.hypot(*qb)
        qa, qb = [x / na for x in qa], [x / nb for x in qb]
        # Chord distance is stable near zero and invariant to quaternion sign.
        chord = min(math.dist(qa, qb), math.dist(qa, [-x for x in qb]))
        angle = min(math.pi, 4 * math.asin(min(1., chord / 2)))
        differences = {"position_l2_m": math.dist(a["position_ned_m"], b["position_ned_m"]),
                       "velocity_l2_m_s": math.dist(a["velocity_ned_m_s"], b["velocity_ned_m_s"]),
                       "attitude_geodesic_rad": angle,
                       "angular_velocity_l2_rad_s": math.dist(a["angular_velocity_rad_s"], b["angular_velocity_rad_s"]),
                       "quaternion_norm_error": max(abs(na - 1), abs(nb - 1))}
        within = {k: v <= spec["tolerances"][k] for k, v in differences.items()}
        equal = {k: a[k] == b[k] for k in DISCRETE}
        result[agent] = {"raw_coordinate_differences": differences, "within_tolerances": within,
                         "discrete_equal": equal, "all_numeric_checks_passed": all(within.values()),
                         "all_discrete_checks_passed": all(equal.values())}
    canonical(result)  # Arithmetic overflow must not become a saved measurement.
    return result


def _artifact(root, entry, expected_path):
    exact_keys(entry, {"path", "sha256", "bytes"})
    name = entry["path"]
    if (not isinstance(name, str) or name != expected_path or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts):
        raise ValueError("Unexpected or escaping artifact path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact resolves outside evidence directory")
    integer(entry["bytes"], "artifact bytes")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or _sha(raw) != entry["sha256"]:
        raise ValueError("Artifact hash/length mismatch")
    return raw


def inspect_startup(directory, index):
    """Verify saved bytes and recompute clock gates, not protobuf/CDR semantics.

    All capture artifacts are checked, not just the explicitly selected row.
    Hash binding proves consistency with the supplied manifest, not authenticity.
    """
    integer(index, "selected index")
    root = Path(directory)
    manifest_raw, rows_raw = (root / "manifest.json").read_bytes(), (root / "samples.jsonl").read_bytes()
    manifest = _decode(manifest_raw)
    if manifest.get("schema") not in {"aerowam-disarmed-startup-evidence/1-draft",
                                       "aerowam-disarmed-startup-evidence/2-draft"}:
        raise ValueError("Unsupported saved startup evidence")
    scene = manifest["scene_sha256"]
    if not isinstance(scene, str) or len(scene) != 64 or any(c not in "0123456789abcdef" for c in scene):
        raise ValueError("Scene SHA256 required")
    for key in ("complete", "start_matched", "full_state_restored", "exposure_state_aligned", "formal_benchmark_result"):
        if type(manifest[key]) is not bool or (key != "complete" and manifest[key]):
            raise ValueError("Unsupported startup evidence claim")
    if (not isinstance(manifest["missing_state"], list)
            or not set(MISSING_STATE).issubset(manifest["missing_state"])):
        raise ValueError("Startup evidence cannot claim complete state coverage")
    for key in ("requested_frames", "captured_frames"):
        integer(manifest[key], key)
    rows = [_decode(line) for line in rows_raw.splitlines() if line.strip()]
    if (not 2 <= manifest["requested_frames"] <= 20 or len(rows) != manifest["captured_frames"]
            or not 0 < len(rows) <= manifest["requested_frames"] or index >= len(rows)):
        raise ValueError("Capture count/selected index mismatch")
    count = 0
    for n, row in enumerate(rows):
        integer(row["index"], "row index")
        if row["index"] != n:
            raise ValueError("Nonsequential evidence rows")
        integer(row["image_stamp_ns"], "image stamp")
        _state(row["state"])
        if n and row["image_stamp_ns"] - rows[n - 1]["image_stamp_ns"] < 250_000_000:
            raise ValueError("Nonadvancing or underspaced image stamp")
        if n and any(row["state"][a]["timestamp_sample_us"] <= rows[n - 1]["state"][a]["timestamp_sample_us"]
                     for a in AGENT_IDS):
            raise ValueError("Nonadvancing odometry stamp")
        expected = {f"{a}_{suffix}" for a in AGENT_IDS
                    for suffix in ("rgb.pb", "labels.pb", "odometry.cdr", "status.cdr")}
        exact_keys(row["artifacts"], expected)
        for name, entry in row["artifacts"].items():
            _artifact(root, entry, f"frame_{n:03d}/{name}")
            count += 1
    local = analyze([r["host_clocks"] for r in rows]) if len(rows) >= 2 else {"issues": ["insufficient_samples"]}
    session, clock_hash = None, None
    issues = []
    if local["issues"]:
        issues.append("capture_clock_quality_failed")
    if not manifest["complete"]:
        issues.append("source_capture_not_complete")
    if manifest["schema"].endswith("/2-draft"):
        saved = manifest["session_clock_quality"]
        raw = _artifact(root.parent, saved["artifact"], "startup_host_clocks.json")
        clock_hash, clock = _sha(raw), _decode(raw)
        if clock["schema"] != PROBE_SCHEMA:
            raise ValueError("Unexpected session clock artifact")
        session = session_quality(clock["samples"], clock["environment"], clock["environment_after"],
                                  clock["session_quality"]["monitor_errors"])
        if (session != clock["session_quality"] or clock["report"] != session["clock_report"]
                or session != {k: v for k, v in saved.items() if k != "artifact"}):
            issues.append("saved_session_quality_inconsistent")
        if not session["quality_passed"]:
            issues.append("session_clock_quality_failed")
        if clock["interrupted"] is not False:
            issues.append("session_clock_record_interrupted")
        if manifest.get("capture_complete") is not True or manifest.get("session_completed") is not True:
            issues.append("capture_or_session_incomplete")
        # Clock samples must cover this capture, not an unrelated stable window.
        times = [c["sample_monotonic_ns"] for c in clock["samples"]]
        if (not times or not all(times[0] <= r["host_clocks"]["sample_monotonic_ns"] <= times[-1] for r in rows)
                or any(not clock["environment"].get(key)
                       or clock["environment"][key] != manifest.get("environment", {}).get(key)
                       or clock["environment"][key] != manifest.get("environment_after", {}).get(key)
                       for key in ("boot_id", "clocksource"))):
            issues.append("capture_session_clock_binding_unverified")
    else:
        issues.append("whole_session_clock_quality_unavailable")
    return {"source": {"schema": manifest["schema"], "manifest_sha256": _sha(manifest_raw),
                       "samples_sha256": _sha(rows_raw), "session_clocks_sha256": clock_hash,
                       "selected_index": index, "image_stamp_ns": rows[index]["image_stamp_ns"],
                       "captured_frames": len(rows), "verified_artifact_files": count,
                       "artifact_semantics_decoded": False, "scene_sha256": scene},
            "state": rows[index]["state"], "capture_clock_report": local,
            "session_clock_quality": session, "quality_issues": issues}


def compare_startup(left_directory, left_index, right_directory, right_index, spec):
    spec = _decode(canonical(spec))
    validate_spec(spec)
    left, right = inspect_startup(left_directory, left_index), inspect_startup(right_directory, right_index)
    diagnostics = compare_states(left["state"], right["state"], spec)
    same_selection = (left["source"]["manifest_sha256"] == right["source"]["manifest_sha256"]
                      and left["source"]["samples_sha256"] == right["source"]["samples_sha256"]
                      and left_index == right_index)
    return {"schema": REPORT_SCHEMA, "purpose": PURPOSE, "spec_sha256": _sha(canonical(spec)),
            "spec": _decode(canonical(spec)), "left": {k: v for k, v in left.items() if k != "state"},
            "right": {k: v for k, v in right.items() if k != "state"},
            "scene_hash_equal": left["source"]["scene_sha256"] == right["source"]["scene_sha256"],
            "same_selected_source": same_selection, "per_vehicle": diagnostics,
            "all_numeric_checks_passed": all(d["all_numeric_checks_passed"] for d in diagnostics.values()),
            "all_discrete_checks_passed": all(d["all_discrete_checks_passed"] for d in diagnostics.values()),
            "common_coordinate_reference_verified": False, "physical_state_comparable": False,
            "missing_requirements": list(MISSING_REQUIREMENTS), "status": "ineligible_start_evidence",
            "start_matched": False, "candidate_outcomes_emitted": False, "formal_benchmark_result": False,
            "interpretation": "Raw coordinate similarity only. No origin fitting, exposure alignment, branch matching, independent repeat count or transfer score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("left", "right", "spec", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("left-index", "right-index"):
        parser.add_argument("--" + key, required=True, type=int)
    args = parser.parse_args()
    result = compare_startup(args.left, args.left_index, args.right, args.right_index,
                             _decode(args.spec.read_bytes()))
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "start_matched": False,
                      "all_numeric_checks_passed": result["all_numeric_checks_passed"], "output": str(args.output)}))
    raise SystemExit(2)  # A written diagnostic is never a candidate-start success.


if __name__ == "__main__":
    main()
