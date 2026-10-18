"""Length-prefixed local policy client with a hard inference timeout."""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from typing import Any

from .protocol import ModelJointAction, ModelObservation


class PolicyTimeoutError(TimeoutError):
    pass


def _remaining(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise socket.timeout("policy deadline expired")
    return remaining


def _receive_exact(connection: socket.socket, size: int, deadline: float | None = None) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            connection.settimeout(_remaining(deadline))
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("policy server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


class PolicyClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 0.4) -> None:
        host, separator, port = endpoint.rpartition(":")
        if not separator or not host:
            raise ValueError("policy endpoint must be HOST:PORT")
        self.host, self.port, self.timeout_s = host, int(port), float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("policy timeout must be positive and finite")

    def infer(self, observation: ModelObservation) -> tuple[ModelJointAction, float]:
        response, latency_ms = self.request(observation.canonical_bytes())
        return ModelJointAction.from_wire(response), latency_ms

    def infer_grounded(self, observation: ModelObservation):
        """Opt-in evidence response; legacy infer() semantics are unchanged."""
        from .grounded_interface import GroundedAction, validate_observed_reports
        response, latency_ms = self.request(observation.canonical_bytes())
        result = GroundedAction.from_wire(response)
        validate_observed_reports(result, observation)
        return result, latency_ms

    def request(self, payload: bytes) -> tuple[Any, float]:
        if len(payload) > 16_000_000:
            raise ValueError("policy request exceeds 16 MB")
        started = time.perf_counter()
        deadline = started + self.timeout_s
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
                connection.settimeout(_remaining(deadline))
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                size = struct.unpack("!I", _receive_exact(connection, 4, deadline))[0]
                if size > 1_000_000:
                    raise ValueError("policy response exceeds 1 MB")
                response: Any = json.loads(_receive_exact(connection, size, deadline))
                _remaining(deadline)
                if isinstance(response, dict) and response.get("error"):
                    raise RuntimeError(
                        f"policy server {response.get('error')}: {response.get('message', '')}"
                    )
        except socket.timeout as exc:
            raise PolicyTimeoutError(f"policy inference exceeded {self.timeout_s * 1000:.0f} ms") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return response, latency_ms
