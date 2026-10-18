"""Serve an explicitly chosen external adapter, without loading built-in weights.

Factory signature: factory() -> callable(wire_request) -> wire_response.
Use DecisionAdapter(predictor) for the draft candidate prediction protocol.
Old live policies return ModelJointAction.to_wire(), preserving the v1 runner.
GroundedAdapter(policy) returns opt-in action/report envelopes; its flight
runner is separate and pending. Never send these envelopes to legacy infer().
--manifest opts into verified entry-source loading and a one-episode diagnostic
service; it is not an admission certificate for model evaluation or flight.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .policy_server import ReusableThreadingTCPServer, _Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a user-selected model adapter factory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--factory", help="importable.module:function (legacy, no identity binding)")
    source.add_argument("--manifest", type=Path, help="Opt-in verified entry-source diagnostic mode")
    parser.add_argument("--entry-artifact", help="Manifest adapter_code ID defining the factory")
    parser.add_argument("--run-dir", type=Path, help="New private runtime record directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.manifest is not None:
        if not args.entry_artifact or args.run_dir is None:
            parser.error("--manifest requires --entry-artifact and --run-dir")
        from .adapter_runtime import RegisteredRuntime, RegisteredHandler
        runtime = RegisteredRuntime(args.manifest, args.entry_artifact, args.run_dir)
        try:
            with ReusableThreadingTCPServer((args.host, args.port), RegisteredHandler) as server:
                server.policy = runtime
                runtime.event("listening", address=list(server.server_address))
                print(f"serving registered diagnostic on {server.server_address}", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        finally:
            runtime.close()
        return
    if args.entry_artifact or args.run_dir is not None:
        parser.error("--entry-artifact/--run-dir require --manifest")
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
