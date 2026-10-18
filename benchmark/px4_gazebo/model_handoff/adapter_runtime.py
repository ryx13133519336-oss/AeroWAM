"""Opt-in, one-episode diagnostic service loaded from verified entry source.

Entry bytes are compiled explicitly, bypassing stale .pyc and imported factory
aliases. This does not authenticate dependencies, weight consumption or GPU
execution, and cannot cancel a Python backend that exceeds its deadline.
"""

from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import socketserver
import struct
import sys
import threading
import time
import os

from .adapter_manifest import audit, _load, check_request, check_saved_io
from .policy_client import _receive_exact, _remaining

SCHEMA = "aerowam-adapter-runtime/1-draft"
DEADLINE_S = .4


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class RegisteredRuntime:
    def __init__(self, manifest_path, entry_artifact, run_dir):
        self.path = Path(manifest_path).resolve()
        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.directory / "events.jsonl").open("xb")
        self._event_lock, self._call_lock = threading.Lock(), threading.Lock()
        self.exchange_lock = threading.Lock()
        self.closed = self.faulted = False
        self.episode_id = None
        self.backend_active = False
        self.policy = None
        self._module_name = self._module = None
        try:
            self.event("starting", pid=os.getpid())
            before = audit(self.path)
            self.manifest = before["declared_method"]
            self._save("preflight.json", before)
            entry = next((a for a in self.manifest["artifacts"] if a["id"] == entry_artifact), None)
            if entry is None or entry["kind"] != "adapter_code":
                raise ValueError("Select a declared adapter_code entry artifact")
            entry_path = (self.path.parent / entry["path"]).resolve()
            if entry_path.suffix != ".py":
                raise ValueError("Registered entry must be Python source (.py)")
            source = entry_path.read_bytes()
            if hashlib.sha256(source).hexdigest() != entry["sha256"]:
                raise ValueError("Entry source changed after preflight")
            with (self.directory / "entry_source.py").open("xb") as stream:
                stream.write(source)
            module_name, function_name = self.manifest["factory"].split(":")
            # Parent/dependency imports follow normal Python rules. Their complete
            # identity and access restrictions are explicitly NOT certified here.
            parent = module_name.rpartition(".")[0]
            if parent:
                importlib.import_module(parent)
            if module_name in sys.modules:
                raise ValueError("Entry module already imported; use a fresh service process")
            spec = importlib.util.spec_from_file_location(module_name, entry_path)
            if spec is None:
                raise ValueError("Cannot construct entry module")
            module = importlib.util.module_from_spec(spec)
            self._module_name, self._module = module_name, module
            sys.modules[module_name] = module
            started = time.perf_counter()
            exec(compile(source, str(entry_path), "exec", dont_inherit=True), module.__dict__)
            factory = getattr(module, function_name)
            if (not inspect.isfunction(factory) or factory.__module__ != module_name
                    or Path(factory.__code__.co_filename).resolve() != entry_path):
                raise ValueError("Factory must be defined in the selected entry source, not re-exported")
            policy = factory()
            if not callable(policy):
                raise TypeError("Factory must return a callable")
            elapsed_ms = (time.perf_counter() - started) * 1000
            after = audit(self.path)
            if before != after:
                raise ValueError("Manifest or declared artifacts changed during factory initialization")
            self.policy = policy
            packages = {}
            for name in ("Pillow", "torch", "transformers", "numpy"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            self._save("startup.json", {"schema": SCHEMA, "method_id": self.manifest["method_id"],
                "manifest_sha256": before["manifest_sha256"], "entry_artifact": entry_artifact,
                "entry_path": str(entry_path), "compiled_entry_sha256": entry["sha256"],
                "entry_source_compiled": True, "factory_returned_callable": True,
                "listed_artifacts_verified_after_factory": True, "factory_initialization_host_ms": elapsed_ms,
                "pid": os.getpid(), "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "package_metadata": packages,
                "online_deadline_ms": 400, "episode_scope": "one_episode_per_process",
                "backend_cancellation_supported": False, "weight_consumption_verified": False,
                "dependency_completeness_verified": False, "gpu_execution_verified": False,
                "access_isolation_verified": False, "end_to_end_deadline_verified": False,
                "role_free_execution_verified": False, "prediction_quality_verified": False,
                "flight_ready": False, "formal_benchmark_result": False})
            self.event("ready_for_diagnostic_requests")
        except BaseException as exc:
            self.faulted = True
            try:
                self.event("startup_failed", error=type(exc).__name__, detail=str(exc))
            finally:
                self.close()
            raise

    def _save(self, name, value):
        with (self.directory / name).open("xb") as stream:
            stream.write(_json_bytes(value) + b"\n")

    def event(self, kind, **values):
        with self._event_lock:
            self._events.write(_json_bytes({"event": kind, "host_monotonic_ns": time.monotonic_ns(), **values}) + b"\n")
            self._events.flush()

    def poison(self, reason):
        self.faulted = True
        self.event("service_faulted", reason=reason)

    def __call__(self, payload, *, deadline=None):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Adapter busy; concurrent requests are rejected, not queued")
        invoked = False
        started = time.perf_counter()
        deadline = started + DEADLINE_S if deadline is None else min(deadline, started + DEADLINE_S)
        try:
            if self.closed or self.faulted:
                raise RuntimeError("Adapter unavailable; start a fresh process")
            original = deepcopy(payload)
            parsed = check_request(self.manifest["interface"], original)
            episode = parsed.history[0].episode_id if self.manifest["interface"] == "candidate_prediction" else parsed.episode_id
            if self.episode_id is not None and episode != self.episode_id:
                raise ValueError("Cross-episode reuse refused; start a fresh service process")
            _remaining(deadline)
            request_hash = hashlib.sha256(_json_bytes(original)).hexdigest()
            self.event("call_started", episode_id=episode, request_sha256=request_hash,
                       request_hash_encoding="sorted_compact_json_utf8")
            _remaining(deadline)
            self.episode_id, self.backend_active = episode, True
            invoked = True
            response = self.policy(deepcopy(original))
            if self.closed or self.faulted:
                raise RuntimeError("Service closed or faulted during the backend call")
            _remaining(deadline)
            # Validate against the original request, even if the adapter mutates
            # its own copy of camera stamps, candidates or language.
            encoded = _json_bytes(response)
            if len(encoded) > 1_000_000:
                raise ValueError("Adapter response exceeds 1 MB")
            result = _load(encoded)
            check_saved_io(self.manifest["interface"], original, result)
            _remaining(deadline)
            self.event("call_contract_verified", response_sha256=hashlib.sha256(encoded).hexdigest(),
                       response_hash_encoding="sorted_compact_json_utf8", host_elapsed_ms=(time.perf_counter()-started)*1000)
            _remaining(deadline)
            return result
        except BaseException as exc:
            if invoked or isinstance(exc, OSError):
                self.faulted = True  # State may have changed; never reuse it.
            if not self._events.closed:
                self.event("call_failed", error=type(exc).__name__, detail=str(exc),
                           backend_invoked=invoked, service_faulted=self.faulted)
            raise
        finally:
            self.backend_active = False
            self._call_lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.event("service_closed", faulted=self.faulted, backend_active_at_close=self.backend_active,
                       backend_cancelled=False, process_exit_verified=False)
        finally:
            with self._event_lock:
                self._events.close()
            if self._module_name and sys.modules.get(self._module_name) is self._module:
                del sys.modules[self._module_name]


class RegisteredHandler(socketserver.BaseRequestHandler):
    def handle(self):
        deadline = time.perf_counter() + DEADLINE_S
        returned = acquired = entered_runtime = False
        try:
            acquired = self.server.policy.exchange_lock.acquire(blocking=False)
            size = struct.unpack("!I", _receive_exact(self.request, 4, deadline))[0]
            if size > 16_000_000:
                raise ValueError("Request exceeds 16 MB")
            raw = _receive_exact(self.request, size, deadline)
            # Decide admission once, before receiving. For a refused exchange,
            # consume its bounded frame before sending the error: closing with
            # unread input can reset TCP and discard the busy response on Windows.
            # This never waits for the backend lock or admits a formerly busy
            # request after that lock becomes free; the same 400ms deadline holds.
            if not acquired:
                raise RuntimeError("Adapter exchange busy; request rejected, not queued")
            payload = _load(raw)
            entered_runtime = True
            result = self.server.policy(payload, deadline=deadline)
            returned = True
            encoded = _json_bytes(result)
            self.request.settimeout(_remaining(deadline))
            self.request.sendall(struct.pack("!I", len(encoded)) + encoded)
            _remaining(deadline)
        except Exception as exc:
            if returned:
                self.server.policy.poison(f"transport_or_call:{type(exc).__name__}")
            if not self.server.policy._events.closed:
                self.server.policy.event("transport_failed", error=type(exc).__name__, detail=str(exc),
                                         entered_runtime=entered_runtime, response_returned=returned)
            # A client can time out before model completion. We cannot cancel an
            # arbitrary backend; fault this diagnostic service instead of reuse.
            try:
                encoded = _json_bytes({"error": type(exc).__name__, "message": str(exc)})
                self.request.settimeout(_remaining(deadline))
                self.request.sendall(struct.pack("!I", len(encoded)) + encoded)
            except (OSError, TimeoutError):
                pass
        finally:
            if acquired:
                self.server.policy.exchange_lock.release()
