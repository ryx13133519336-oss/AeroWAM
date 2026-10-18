"""Private receive-time evidence, not a simulator snapshot or match certificate."""

import hashlib
import json
import math
from pathlib import Path

from .clock_probe import ANALYSIS_PROFILE, LIMIT_NS, analyze as analyze_clocks, environment


def validate_mode(count, *, execute_flight, control_link_check, policy, teammate_response):
    if isinstance(count, bool) or not isinstance(count, int) or count not in (0, *range(2, 21)):
        raise ValueError("--startup-evidence-frames must be 0 (disabled) or 2..20")
    if count and (execute_flight or control_link_check or policy != "rgb_scan" or teammate_response != "nominal"):
        raise ValueError("Startup evidence requires rgb_scan, nominal response, no flight and no command check")


def _vector(value, length):
    result = [float(v) for v in value]
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite or malformed startup state")
    return result


class StartupEvidence:
    """Append-only raw evidence. All artifacts stay in the private directory."""

    def __init__(self, directory, count, scene_sha256):
        validate_mode(count, execute_flight=False, control_link_check=False,
                      policy="rgb_scan", teammate_response="nominal")
        if count == 0:
            raise ValueError("Evidence capture must be enabled")
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.count, self.rows = count, []
        self.manifest = {
            "schema": "aerowam-disarmed-startup-evidence/1-draft",
            "scene_sha256": scene_sha256, "requested_frames": count,
            "sample_min_sim_spacing_ns": 250_000_000, "collection_wall_timeout_s": 15.,
            "receive_age_limits_s": {"status": 2., "odometry": .5, "camera": .25, "clock": .5},
            "image_clock_lag_limit_ns": 200_000_000,
            "sampled_host_clock_offset_change_limit_ns": LIMIT_NS,
            "clock_analysis_profile": ANALYSIS_PROFILE, "environment": environment(),
            "sampling": "latest_received_state_with_same_stamp_dual_rgb_and_labels",
            "complete": False, "captured_frames": 0, "start_matched": False,
            "full_state_restored": False, "exposure_state_aligned": False,
            "role_assignment": "none_no_task_actions_executed",
            "missing_state": ["physics_internal_state", "controller_and_estimator_internal_state",
                              "rng_state", "pending_control_and_sensor_queues", "task_and_teammate_state"],
            "formal_benchmark_result": False,
        }
        self._write_manifest()

    def _write_manifest(self):
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False), encoding="utf-8")

    def add(self, *, odometry, status, received, clock, cameras, serialize_ros, host_clock, now):
        """Inputs must be atomically frozen references; no cross-clock subtraction."""
        if len(self.rows) >= self.count:
            raise ValueError("Unexpected extra evidence frame")
        if set(cameras) != {"uav_0", "uav_1"}:
            raise ValueError("Both cameras required")
        stamps = {entry["stamp_ns"] for entry in cameras.values()}
        if len(stamps) != 1:
            raise ValueError("Dual camera stamps differ")
        stamp = stamps.pop()
        if (not isinstance(stamp, int) or stamp < 0
                or not 0 <= clock["stamp"] - stamp <= 200_000_000
                or not 0 <= now - clock["received"] <= .5):
            raise ValueError("Stale or invalid simulation clock")
        if self.rows and stamp - self.rows[-1]["image_stamp_ns"] < 250_000_000:
            raise ValueError("Evidence camera stamp repeated or spacing too short")
        row = {"index": len(self.rows), "image_stamp_ns": stamp,
               "sim_clock_ns": clock["stamp"], "clock_received_monotonic_s": clock["received"],
               "host_clocks": host_clock, "state": {}, "artifacts": {}}
        # Validate serializability before any artifact is created.
        json.dumps(row, allow_nan=False)
        blobs = {}
        for i in (0, 1):
            agent = f"uav_{i}"
            odom, state, camera = odometry[i], status[i], cameras[agent]
            if int(state.arming_state) != 1 or state.failsafe:
                raise ValueError("Evidence requires two disarmed healthy vehicles")
            for kind, limit in (("status", 2.), ("odometry", .5)):
                if not 0 <= now - received[kind, i] <= limit:
                    raise ValueError(f"Stale startup {kind}")
            if int(odom.pose_frame) != 1 or int(odom.velocity_frame) != 1:
                raise ValueError("Known NED frames required")
            if self.rows and int(odom.timestamp_sample) <= self.rows[-1]["state"][agent]["timestamp_sample_us"]:
                raise ValueError("Odometry sample did not advance")
            for kind in ("rgb", "labels"):
                message, arrival = camera[kind]
                message_stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nsec)
                if message_stamp != stamp or not 0 <= now - arrival <= .25:
                    raise ValueError("Stale or mismatched camera evidence")
                blobs[f"{agent}_{kind}.pb"] = message.SerializeToString()
            blobs[f"{agent}_odometry.cdr"] = bytes(serialize_ros(odom))
            blobs[f"{agent}_status.cdr"] = bytes(serialize_ros(state))
            row["state"][agent] = {
                "position_ned_m": _vector(odom.position, 3), "velocity_ned_m_s": _vector(odom.velocity, 3),
                "quaternion_wxyz": _vector(odom.q, 4), "angular_velocity_rad_s": _vector(odom.angular_velocity, 3),
                "pose_frame": int(odom.pose_frame), "velocity_frame": int(odom.velocity_frame),
                "timestamp_us": int(odom.timestamp), "timestamp_sample_us": int(odom.timestamp_sample),
                "reset_counter": int(odom.reset_counter), "quality": int(odom.quality),
                "arming_state": int(state.arming_state), "nav_state": int(state.nav_state),
                "failsafe": bool(state.failsafe), "pre_flight_checks_pass": bool(state.pre_flight_checks_pass),
                "received_monotonic_s": {kind: received[kind, i] for kind in ("status", "odometry")},
                "camera_received_monotonic_s": {kind: camera[kind][1] for kind in ("rgb", "labels")},
            }
        # Validate before creating the frame directory. Partial IO failures remain
        # visible on disk and cannot acquire a completed manifest.
        frame_dir = self.directory / f"frame_{len(self.rows):03d}"
        frame_dir.mkdir(exist_ok=False)
        for name, blob in blobs.items():
            with (frame_dir / name).open("xb") as stream:
                stream.write(blob)
            row["artifacts"][name] = {"path": f"{frame_dir.name}/{name}",
                                     "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        with (self.directory / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        self.manifest["captured_frames"] = len(self.rows)
        self._write_manifest()

    def finish(self):
        if len(self.rows) != self.count:
            raise ValueError("Incomplete startup evidence")
        self.manifest["environment_after"] = environment()
        self.manifest["sampled_host_clock_report"] = analyze_clocks([r["host_clocks"] for r in self.rows])
        if self.manifest["sampled_host_clock_report"]["issues"]:
            self._write_manifest()
            raise ValueError("Sampled host clocks inconsistent during evidence capture")
        self.manifest["complete"] = True
        self.manifest["span_sim_s"] = (self.rows[-1]["image_stamp_ns"] - self.rows[0]["image_stamp_ns"]) / 1e9
        self._write_manifest()
        return dict(self.manifest)
