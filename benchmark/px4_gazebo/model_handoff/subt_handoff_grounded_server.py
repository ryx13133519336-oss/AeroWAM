"""Serve a grounded policy through the SubT Core JSON wire protocol.

The factory must be ``MODULE:FUNCTION`` and return a callable that accepts the
legacy ``ModelObservation`` and returns ``GroundedAction``. The default bridge
holds both proxy cameras; ENU motion is an explicit diagnostic opt-in.
"""

from __future__ import annotations

import argparse
import importlib
import json
import socketserver
import struct
import time
from typing import Any

from .subt_handoff_grounded_bridge import make_core_policy


def _receive_exact(connection: Any, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("client closed request")
        data.extend(chunk)
    return bytes(data)


class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        started = time.perf_counter()
        try:
            size = struct.unpack("!I", _receive_exact(self.request, 4))[0]
            if size > 16_000_000:
                raise ValueError("request exceeds 16 MB")
            payload = json.loads(_receive_exact(self.request, size))
            response = self.server.policy(payload)  # type: ignore[attr-defined]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms > 400.0:
                raise TimeoutError("grounded policy exceeded 400 ms")
        except Exception as exc:
            response = {"error": type(exc).__name__, "message": str(exc)}
        encoded = json.dumps(response, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(encoded) > 1_000_000:
            encoded = b'{"error":"ValueError","message":"response exceeds 1 MB"}'
        self.request.sendall(struct.pack("!I", len(encoded)) + encoded)


def load_factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("factory must be MODULE:FUNCTION")
    factory = getattr(importlib.import_module(module_name), function_name)
    policy = factory()
    if not callable(policy):
        raise TypeError("factory must return a callable")
    return policy


def serve(factory_spec: str, *, host: str = "127.0.0.1", port: int = 8765,
          allow_proxy_enu_actions: bool = False) -> None:
    grounded_policy = load_factory(factory_spec)
    policy = make_core_policy(grounded_policy, allow_proxy_enu_actions=allow_proxy_enu_actions)
    with ReusableServer((host, port), Handler) as server:
        server.policy = policy
        print(json.dumps({"status": "listening", "host": host, "port": port,
                          "factory": factory_spec,
                          "allow_proxy_enu_actions": allow_proxy_enu_actions}), flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="MODULE:FUNCTION returning a grounded policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-proxy-enu-actions", action="store_true",
                        help="Explicitly map legacy body-FLU deltas to proxy ENU; diagnostic only")
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1024 <= args.port <= 65535:
        parser.error("Only a loopback endpoint on port 1024..65535 is supported")
    serve(args.factory, host=args.host, port=args.port,
          allow_proxy_enu_actions=args.allow_proxy_enu_actions)


if __name__ == "__main__":
    main()
