"""Length-prefixed local policy client with a hard inference timeout."""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any

from .protocol import ModelJointAction, ModelObservation


class PolicyTimeoutError(TimeoutError):
    pass


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
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

    def infer(self, observation: ModelObservation) -> tuple[ModelJointAction, float]:
        payload = observation.canonical_bytes()
        started = time.perf_counter()
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                size = struct.unpack("!I", _receive_exact(connection, 4))[0]
                if size > 1_000_000:
                    raise ValueError("policy response exceeds 1 MB")
                response: Any = json.loads(_receive_exact(connection, size))
                if isinstance(response, dict) and response.get("error"):
                    raise RuntimeError(
                        f"policy server {response.get('error')}: {response.get('message', '')}"
                    )
        except socket.timeout as exc:
            raise PolicyTimeoutError(f"policy inference exceeded {self.timeout_s * 1000:.0f} ms") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ModelJointAction.from_wire(response), latency_ms
