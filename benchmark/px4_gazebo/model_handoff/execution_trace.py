"""Opt-in trace of published PX4 commands, never asserted motor execution."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from .protocol import AGENT_IDS, assert_truth_free

TRACE_SCHEMA = "aerowam-published-controls/1"
TIMED_TRACE_SCHEMA = "aerowam-published-controls/2"
BOUNDED_TRACE_SCHEMA = "aerowam-published-controls/3"
TIMED_SCHEMAS = {TIMED_TRACE_SCHEMA, BOUNDED_TRACE_SCHEMA}


def host_clock_pair() -> dict:
    """Bracket a UNIX read; do not assume two sequential reads are atomic."""
    before = time.monotonic_ns()
    unix = time.time_ns()
    after = time.monotonic_ns()
    return {"monotonic_ns": before, "monotonic_after_ns": after, "unix_ns": unix}


def _validate_clock_bound(row: Mapping, before_key: str, after_key: str) -> None:
    before, after = row[before_key], row[after_key]
    if not isinstance(after, int) or after < before:
        raise ValueError("Invalid host clock read bracket")


def trace_clocks(timed: bool) -> dict:
    return {"interval": "host_monotonic_ns", "command": "host_unix_us",
            "odometry": "dds_received_timestamp_us_conversion_unverified" if timed else "px4_timestamp_us_unaligned",
            "image": "ros_image_stamp_ns_unaligned"}


def odometry_snapshot(message: Any, *, received_monotonic_s: float) -> dict:
    """Keep DDS timestamp values and PX4 frames; do not apply another offset."""
    result = {
        "position": [float(v) for v in message.position],
        "velocity": [float(v) for v in message.velocity],
        "quaternion_wxyz": [float(v) for v in message.q],
        "pose_frame": int(message.pose_frame), "velocity_frame": int(message.velocity_frame),
        "timestamp_us": int(message.timestamp), "timestamp_sample_us": int(message.timestamp_sample),
        "received_monotonic_ns": int(received_monotonic_s * 1e9),
    }
    _validate_odometry(result)
    return result


def _finite_vector(value: Any, length: int) -> None:
    if len(value) != length or not all(math.isfinite(float(v)) for v in value):
        raise ValueError("Trace vector has invalid size/non-finite values")


def _validate_odometry(row: Mapping) -> None:
    for name, size in (("position", 3), ("velocity", 3), ("quaternion_wxyz", 4)):
        _finite_vector(row[name], size)
    for key in ("timestamp_us", "timestamp_sample_us", "received_monotonic_ns"):
        if int(row[key]) < 0:
            raise ValueError("Negative odometry timestamp")
    if row["pose_frame"] not in {1, 2} or row["velocity_frame"] not in {1, 2, 3}:
        raise ValueError("Undefined odometry coordinate frame")
    norm = sum(float(v) ** 2 for v in row["quaternion_wxyz"])
    if not math.isclose(norm, 1.0, abs_tol=0.02):
        raise ValueError("Invalid odometry quaternion")


class ExecutionTrace:
    """Buffer RUN publications and flush after landing, keeping tick I/O small.

    An interval spans successive publication calls or the transition to landing.
    It is not an acknowledged command holding time or motor actuation.
    """

    def __init__(self, public_dir: Path, episode_id: str, *, record_timing: bool = False,
                 record_clock_bounds: bool = False) -> None:
        if record_clock_bounds and not record_timing:
            raise ValueError("Clock read bounds require timing evidence")
        self.path = public_dir / "published_controls.jsonl"
        self.episode_id = episode_id
        self.pending: dict[str, dict] = {}
        self.rows: list[dict] = []
        self.samples: list[dict] = []
        self.finished = False
        self.reason: str | None = None
        self.flushed = False
        self.error: str | None = None
        self.record_timing = record_timing
        self.record_clock_bounds = record_clock_bounds
        self.sim_clock: list[dict] = []
        self.timesync: dict[str, list[dict]] = {agent: [] for agent in AGENT_IDS}

    def clock_received(self, *, stamp_ns: int, monotonic_ns: int, unix_ns: int,
                       monotonic_after_ns: int | None = None) -> None:
        if self.record_timing and not self.finished:
            # Keep approximately 20 Hz receive evidence, plus every detected
            # backward clock jump. No transport/file I/O in callbacks.
            if self.sim_clock and (stamp_ns >= self.sim_clock[-1]["stamp_ns"]
                                   and monotonic_ns - self.sim_clock[-1]["received_monotonic_ns"] < 50_000_000):
                return
            self.sim_clock.append({"stamp_ns": stamp_ns, "received_monotonic_ns": monotonic_ns,
                                   "received_unix_ns": unix_ns})
            if self.record_clock_bounds:
                self.sim_clock[-1]["clock_read_end_monotonic_ns"] = monotonic_after_ns

    def timesync_received(self, agent_id: str, message: Any, *, monotonic_ns: int, unix_ns: int,
                          monotonic_after_ns: int | None = None) -> None:
        if self.record_timing and not self.finished:
            self.timesync[agent_id].append({
                key: int(getattr(message, key)) for key in (
                    "timestamp", "source_protocol", "remote_timestamp", "observed_offset",
                    "estimated_offset", "round_trip_time")
            } | {"received_monotonic_ns": monotonic_ns, "received_unix_ns": unix_ns})
            if self.record_clock_bounds:
                self.timesync[agent_id][-1]["clock_read_end_monotonic_ns"] = monotonic_after_ns

    def fail(self, error: Exception) -> None:
        # Logging failures must never prevent landing. Validation will reject
        # the incomplete dataset rather than silently training on it.
        self.error = self.error or f"{type(error).__name__}: {error}"

    def anchor(self, *, sample_index: int, protocol_t: int, capture_monotonic_ns: int,
               image_timestamp_ns: Mapping[str, int], timing: Mapping | None = None) -> None:
        if self.finished:
            raise ValueError("Sample after trace finish")
        self.samples.append({"sample_index": sample_index, "protocol_t": protocol_t,
                             "capture_monotonic_ns": capture_monotonic_ns,
                             "image_timestamp_ns": dict(image_timestamp_ns)})
        if self.record_timing:
            if timing is None:
                raise ValueError("Timed trace requires observation timing")
            self.samples[-1]["timing"] = deepcopy(dict(timing))

    def published(self, *, agent_id: str, monotonic_ns: int, command_timestamp_us: int,
                  protocol_t: int | None, body_flu: list[float], velocity_ned: list[float],
                  yaw_rate: float, odometry: Mapping) -> None:
        if self.finished or agent_id not in AGENT_IDS:
            raise ValueError("Publication after finish or invalid agent")
        _finite_vector(body_flu, 4)
        _finite_vector(velocity_ned, 3)
        _validate_odometry(odometry)
        if not math.isfinite(yaw_rate) or command_timestamp_us < 0 or monotonic_ns < 0:
            raise ValueError("Invalid published control")
        self._end(agent_id, monotonic_ns, odometry)
        self.pending[agent_id] = {
            "agent_id": agent_id, "start_monotonic_ns": monotonic_ns,
            "command_timestamp_us": command_timestamp_us, "protocol_t": protocol_t,
            "body_flu_velocity_yaw_rate": list(body_flu),
            "published_ned_velocity": list(velocity_ned), "published_yaw_rate": yaw_rate,
            "start_odometry": deepcopy(dict(odometry)),
        }

    def _end(self, agent_id: str, timestamp: int, observation: Mapping) -> None:
        previous = self.pending.get(agent_id)
        if previous is None:
            return
        if timestamp <= previous["start_monotonic_ns"]:
            raise ValueError("Non-increasing publication clock")
        self.rows.append({**previous, "end_monotonic_ns": timestamp,
                          "duration_s": (timestamp - previous["start_monotonic_ns"]) / 1e9,
                          "end_odometry": deepcopy(dict(observation))})

    def finish(self, timestamp: int, observations: Mapping[str, Mapping], reason: str) -> None:
        if self.finished:
            return
        for agent_id, previous in self.pending.items():
            _validate_odometry(observations[agent_id])
            if timestamp <= previous["start_monotonic_ns"]:
                raise ValueError("Non-increasing finish clock")
        for agent_id in list(self.pending):
            self._end(agent_id, timestamp, observations[agent_id])
        self.pending.clear()
        self.finished = True
        self.reason = reason

    def flush(self) -> None:
        if self.flushed:
            return
        if not self.finished:
            raise ValueError("Cannot flush an unfinished trace")
        header = {
            "schema": (BOUNDED_TRACE_SCHEMA if self.record_clock_bounds else
                       TIMED_TRACE_SCHEMA if self.record_timing else TRACE_SCHEMA),
            "episode_id": self.episode_id, "finished": True,
            "end_reason": self.reason, "semantics": "published_setpoints_not_verified_actuation",
            "clocks": trace_clocks(self.record_timing),
            "sample_anchors": self.samples,
        }
        if self.record_timing:
            header["timing_evidence"] = {"sim_clock": self.sim_clock, "timesync": self.timesync,
                                         "alignment": "unverified_no_offset_applied"}
        if self.error is not None:
            header["finished"] = False
            header["trace_error"] = self.error
        else:
            validate_trace(header, self.rows)
        # No synchronous disk writes in the 20 Hz publication path.
        with self.path.open("x", encoding="utf-8") as output:
            for row in (header, *self.rows):
                output.write(json.dumps(row, allow_nan=False) + "\n")
        self.flushed = True


def validate_trace(header: Mapping, rows: list[dict]) -> dict:
    assert_truth_free(header)
    assert_truth_free(rows)
    timed = header.get("schema") in TIMED_SCHEMAS
    bounded = header.get("schema") == BOUNDED_TRACE_SCHEMA
    if header.get("schema") not in {TRACE_SCHEMA, *TIMED_SCHEMAS} or header.get("finished") is not True:
        raise ValueError("Invalid or unfinished control trace")
    if header.get("semantics") != "published_setpoints_not_verified_actuation":
        raise ValueError("Unverified actuation must not be mislabeled")
    if header.get("trace_error"):
        raise ValueError("Trace recording failed")
    if header.get("clocks") != trace_clocks(timed):
        raise ValueError("Invalid trace clock semantics")
    if not rows:
        raise ValueError("No published RUN commands")
    last = {}
    counts = {agent: 0 for agent in AGENT_IDS}
    for row in rows:
        agent = row["agent_id"]
        if agent not in counts:
            raise ValueError("Unknown trace agent")
        start, end = int(row["start_monotonic_ns"]), int(row["end_monotonic_ns"])
        if start < 0 or end <= start or (agent in last and start != last[agent]):
            raise ValueError("Non-contiguous published command intervals")
        if not math.isclose(float(row["duration_s"]), (end - start) / 1e9, abs_tol=1e-12):
            raise ValueError("Command duration mismatch")
        _finite_vector(row["body_flu_velocity_yaw_rate"], 4)
        _finite_vector(row["published_ned_velocity"], 3)
        if not math.isfinite(float(row["published_yaw_rate"])) or int(row["command_timestamp_us"]) < 0:
            raise ValueError("Invalid published control/time")
        if row["protocol_t"] is not None and (not isinstance(row["protocol_t"], int) or row["protocol_t"] < 0):
            raise ValueError("Invalid controlling policy step")
        for endpoint in ("start", "end"):
            observation = row[f"{endpoint}_odometry"]
            _validate_odometry(observation)
            if observation["received_monotonic_ns"] > row[f"{endpoint}_monotonic_ns"]:
                raise ValueError("Future odometry leaked into interval endpoint")
        last[agent] = end
        counts[agent] += 1
    if not all(counts.values()):
        raise ValueError("Missing one vehicle's commands")
    anchors = header["sample_anchors"]
    if not anchors:
        raise ValueError("No observation anchors")
    for index, sample in enumerate(anchors):
        if sample["sample_index"] != index or sample["capture_monotonic_ns"] < 0 or sample["protocol_t"] < 0:
            raise ValueError("Invalid sample anchor")
        if set(sample["image_timestamp_ns"]) != set(AGENT_IDS):
            raise ValueError("Missing image stamps")
        if any(not isinstance(stamp, int) or stamp < 0 for stamp in sample["image_timestamp_ns"].values()):
            raise ValueError("Invalid image stamp")
        if index and sample["capture_monotonic_ns"] <= header["sample_anchors"][index - 1]["capture_monotonic_ns"]:
            raise ValueError("Non-increasing sample capture clock")
        if index and sample["protocol_t"] <= anchors[index - 1]["protocol_t"]:
            raise ValueError("Non-increasing sample policy step")
        for agent in AGENT_IDS:
            first = next(row["start_monotonic_ns"] for row in rows if row["agent_id"] == agent)
            if not first <= sample["capture_monotonic_ns"] <= last[agent]:
                raise ValueError("Sample outside RUN publication window")
        if timed:
            timing = sample["timing"]
            if not isinstance(timing["capture_unix_ns"], int) or timing["capture_unix_ns"] < 0:
                raise ValueError("Invalid capture UNIX clock")
            if bounded:
                _validate_clock_bound({**timing, "capture_monotonic_ns": sample["capture_monotonic_ns"]},
                                      "capture_monotonic_ns", "clock_read_end_monotonic_ns")
            for name in ("image_received_monotonic_ns", "odometry"):
                if set(timing[name]) != set(AGENT_IDS):
                    raise ValueError("Missing observation timing agent")
            for agent in AGENT_IDS:
                received = timing["image_received_monotonic_ns"][agent]
                if not isinstance(received, int) or not 0 <= received <= sample["capture_monotonic_ns"]:
                    raise ValueError("Future image receive time")
                observation = timing["odometry"][agent]
                _validate_odometry(observation)
                if observation["received_monotonic_ns"] > sample["capture_monotonic_ns"]:
                    raise ValueError("Future sample odometry")
    if timed:
        evidence = header["timing_evidence"]
        if evidence.get("alignment") != "unverified_no_offset_applied" or set(evidence["timesync"]) != set(AGENT_IDS):
            raise ValueError("Invalid timing evidence semantics")
        for samples in (evidence["sim_clock"], *evidence["timesync"].values()):
            previous = -1
            for sample in samples:
                received = sample["received_monotonic_ns"]
                if not isinstance(received, int) or received <= previous or received > max(last.values()):
                    raise ValueError("Invalid clock evidence receive order")
                previous = received
                if bounded:
                    _validate_clock_bound(sample, "received_monotonic_ns", "clock_read_end_monotonic_ns")
                keys = ("stamp_ns", "received_unix_ns") if "stamp_ns" in sample else (
                    "timestamp", "source_protocol", "remote_timestamp", "round_trip_time", "received_unix_ns")
                if any(not isinstance(sample[key], int) or sample[key] < 0 for key in keys):
                    raise ValueError("Invalid clock evidence value")
                if "stamp_ns" not in sample and any(not isinstance(sample[key], int) for key in ("observed_offset", "estimated_offset")):
                    raise ValueError("Invalid signed timesync offset")
    return {"published_intervals": len(rows), "agent_intervals": counts, "samples": len(header["sample_anchors"])}


def validate_execution_file(path: Path, *, samples: list[dict], episode_id: str) -> dict:
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not entries:
        raise ValueError("Empty execution trace")
    header, rows = entries[0], entries[1:]
    summary = validate_trace(header, rows)
    if header["episode_id"] != episode_id or summary["samples"] != len(samples):
        raise ValueError("Execution trace/sample identity mismatch")
    for sample, anchor in zip(samples, header["sample_anchors"]):
        if (sample["sample_index"] != anchor["sample_index"]
                or sample["image_timestamp_ns"] != anchor["image_timestamp_ns"]
                or sample.get("protocol_t") != anchor["protocol_t"]):
            raise ValueError("Execution trace image/sample alignment mismatch")
    return summary
