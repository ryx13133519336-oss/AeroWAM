"""Export or check dual-aircraft warehouse. Never publishes flight commands."""

import argparse
import json
from pathlib import Path

from .warehouse_scene import aircraft_world, sha256


def export(package, models, output):
    tree, resources = aircraft_world(package, models)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tree.write(output / "world.sdf", encoding="utf-8", xml_declaration=True)
    contract = {"schema": "aerowam-warehouse-integration/1-draft", **resources,
                "world_sha256": sha256(output / "world.sdf"),
                "policy_evaluated": False, "motion_commands_sent": 0}
    (output / "scene_contract.private.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-check", action="store_true", help="Start isolated disarmed PX4 20/21; no publishers")
    parser.add_argument("--px4-root", type=Path)
    parser.add_argument("--agent", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.)
    parser.add_argument("--ros-domain-id", type=int, default=79)
    args = parser.parse_args()
    if not args.startup_check:
        if args.gui:
            parser.error("--gui requires --startup-check; exported world can be opened with gz sim")
        result = export(args.package, args.models, args.output_dir)
        print(json.dumps({k: result[k] for k in ("schema", "world_name", "world_sha256", "readiness", "flight_admitted")}, indent=2))
        return
    if args.px4_root is None or args.agent is None:
        parser.error("--startup-check requires --px4-root and --agent")
    from .search_pilot import inspect_agent
    from .annotation_session_smoke import run
    inspect_agent(args.agent)
    result = run(px4_root=args.px4_root, models=args.models, output=args.output_dir,
                 prepared_scene=aircraft_world(args.package, args.models), agent_path=args.agent,
                 timeout_s=args.timeout_s, domain=args.ros_domain_id, udp_only=True, gui=args.gui)
    print(json.dumps({k: result.get(k) for k in ("startup_passed", "flight_ready", "motion_commands_sent",
                                                "pair_counts", "ros_counts", "error", "owned_processes_stopped",
                                                "cleanup_errors")}, indent=2))
    raise SystemExit(0 if result["startup_passed"] and result["owned_processes_stopped"] and not result.get("cleanup_errors") else 2)


if __name__ == "__main__":
    main()
