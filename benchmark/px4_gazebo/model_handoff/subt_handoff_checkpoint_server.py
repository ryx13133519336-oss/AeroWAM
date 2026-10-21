"""Serve the frozen SubT checkpoint over the core length-prefixed JSON wire."""

from __future__ import annotations

import argparse
import json
import socketserver
import struct
from pathlib import Path

from .core._common import canonical
from .subt_handoff_checkpoint_adapter import create_policy


def _receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("client closed request")
        chunks.extend(chunk)
    return bytes(chunks)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            size = struct.unpack("!I", _receive_exact(self.request, 4))[0]
            if size > 16_000_000:
                raise ValueError("request exceeds 16 MB")
            request = json.loads(_receive_exact(self.request, size))
            response = self.server.policy(request)
        except Exception as exc:
            response = {"error": type(exc).__name__, "message": str(exc)}
        payload = canonical(response)
        self.request.sendall(struct.pack("!I", len(payload)) + payload)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    policy = create_policy(args.checkpoint)
    with Server((args.host, args.port), _Handler) as server:
        server.policy = policy
        print(json.dumps({"status": "listening", "host": args.host,
                          "port": args.port, "checkpoint": str(args.checkpoint)}, indent=2), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
