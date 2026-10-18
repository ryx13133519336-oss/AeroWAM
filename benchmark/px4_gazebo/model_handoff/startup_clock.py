"""Receive-only session clock gate. Never changes clocks or starts processes."""

import hashlib
import json
from pathlib import Path
import threading

from .clock_probe import ANALYSIS_PROFILE, PROBE_SCHEMA, analyze, environment, sample_clocks

SCOPE = "before_transport_setup_through_owned_process_cleanup"
INTERVAL_S = .05
MAX_GAP_MS = 250.
MAX_SAMPLES = 10_000


def session_quality(samples, before, after, errors=()):
    """Fixed draft coverage checks in addition to the unchanged 10ms analysis."""
    issues = []
    try:
        report = analyze(samples)
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        report = {"analysis_profile": ANALYSIS_PROFILE, "issues": ["clock_analysis_error"],
                  "error": f"{type(exc).__name__}:{exc}"}
    issues.extend(report["issues"])
    if report.get("max_poll_gap_ms", 0) > MAX_GAP_MS:
        issues.append("clock_poll_gap")
    if any(not report.get("metrics", {}).get(name, {}).get("available", False)
           for name in ("unix_ns", "raw_ns", "boottime_ns")):
        issues.append("required_host_clock_unavailable")
    if any(not before.get(key) or not after.get(key) or before[key] != after[key]
           for key in ("boot_id", "clocksource")):
        issues.append("clock_environment_unverified")
    if errors:
        issues.append("clock_monitor_error")
    return {"schema": "aerowam-startup-session-clock-quality/1-draft", "scope": SCOPE,
            "target_poll_interval_ms": INTERVAL_S * 1000, "max_allowed_poll_gap_ms": MAX_GAP_MS,
            "required_clocks": ["unix_ns", "raw_ns", "boottime_ns"],
            "quality_passed": not issues, "issues": issues, "monitor_errors": list(errors),
            "clock_report": report,
            "interpretation": "Sampled session clock quality only; not exposure alignment, state restoration or task performance."}


class StartupClockMonitor:
    """Bounded memory; synchronous endpoint samples bracket the worker lifetime."""

    def __init__(self, path):
        self.path = Path(path)
        self.samples, self.errors = [], []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = self._stream = self._result = None
        self._before = {}

    def _sample(self):
        value = sample_clocks()
        with self._lock:
            if len(self.samples) >= MAX_SAMPLES:
                raise RuntimeError("Startup clock sample limit reached")
            self.samples.append(value)

    def _worker(self):
        try:
            while not self._stop.wait(INTERVAL_S):
                self._sample()
        except Exception as exc:
            with self._lock:
                self.errors.append(f"{type(exc).__name__}:{exc}")

    def start(self):
        if self._stream is not None:
            raise RuntimeError("Startup clock monitor already started")
        self._stream = self.path.open("xb")  # Reserve; never overwrite evidence.
        try:
            self._before = environment()
            self._sample()  # Must succeed before any session process can start.
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.errors.append(f"start:{type(exc).__name__}:{exc}")
            raise

    def check(self):
        with self._lock:
            if self.errors:
                raise RuntimeError("Startup clock monitor failed: " + str(self.errors))

    def finish(self):
        if self._result is not None:
            return self._result
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=2.)
        try:
            if self._thread is not None and self._thread.is_alive():
                with self._lock:
                    self.errors.append("Clock monitor thread did not stop")
            else:
                try:
                    self._sample()
                except Exception as exc:
                    with self._lock:
                        self.errors.append(f"{type(exc).__name__}:{exc}")
            after = environment()
            with self._lock:
                samples, errors = list(self.samples), list(self.errors)
            quality = session_quality(samples, self._before, after, errors)
            payload = {"schema": PROBE_SCHEMA, "environment": self._before,
                       "environment_after": after, "interrupted": bool(errors),
                       "samples": samples, "report": quality["clock_report"],
                       "session_quality": quality}
            raw = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")
            if self._stream is None:
                raise RuntimeError("Startup clock output was not reserved")
            self._stream.write(raw)
            self._stream.flush()
            self._result = {**quality, "artifact": {"path": self.path.name,
                             "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
            return self._result
        finally:
            if self._stream is not None:
                self._stream.close()


def finalize_startup_clock(monitor, summary, evidence):
    """Called after process cleanup, even for failed startup or teardown."""
    session_ok = (summary.get("category") == "startup_only"
                  and summary.get("owned_processes_stopped") is True
                  and not summary.get("cleanup_errors"))
    try:
        quality = monitor.finish()
    except Exception as exc:
        quality = {"quality_passed": False, "scope": SCOPE,
                   "issues": ["clock_monitor_finalize_error"], "error": f"{type(exc).__name__}:{exc}"}
    summary["startup_clock_quality"] = quality
    summary["startup_receive_quality_passed"] = False
    if evidence is not None:
        try:
            summary["startup_evidence"] = evidence.finish_session(quality, session_succeeded=session_ok)
            summary["startup_receive_quality_passed"] = summary["startup_evidence"]["complete"]
        except Exception as exc:
            summary["startup_evidence_finalize_error"] = f"{type(exc).__name__}:{exc}"
    if summary.get("category") == "startup_only" and not summary["startup_receive_quality_passed"]:
        summary["result_before_startup_clock_gate"] = {key: summary[key] for key in ("category", "reason")}
        summary.update(category="simulator_error", reason="startup_session_quality_failed")
