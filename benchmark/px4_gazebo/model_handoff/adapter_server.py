"""Serve an explicitly chosen external adapter, without loading built-in weights.

Factory signature: factory() -> callable(wire_request) -> wire_response.
Use DecisionAdapter(predictor) for the draft candidate prediction protocol.
Old live policies return ModelJointAction.to_wire(), preserving the v1 runner.
GroundedAdapter(policy) returns opt-in action/report envelopes; its flight
runner is separate and pending. Never send these envelopes to legacy infer().
"""

from __future__ import annotations

import argparse
import importlib

from .policy_server import ReusableThreadingTCPServer, _Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a user-selected model adapter factory")
    parser.add_argument("--factory", required=True, help="importable.module:function (no arguments)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    module, separator, name = args.factory.partition(":")
    if not separator or not module or not name:
        parser.error("factory must be importable.module:function")
    policy = getattr(importlib.import_module(module), name)()
    if not callable(policy):
        raise TypeError("Adapter factory must return a callable")
    with ReusableThreadingTCPServer((args.host, args.port), _Handler) as server:
        server.policy = policy
        print(f"serving adapter {args.factory} on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
