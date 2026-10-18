"""Matched RGB history-information diagnostic; no model, training or flight."""

import argparse
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .search_occlusion_sequence import (
    LAYOUTS, _load_message, inspect as inspect_sequence, run as capture,
    sequence_world, target_poses,
)

SCHEMA = "aerowam-search-history-probe/1-draft"
AGENTS = ("uav_0", "uav_1")
COLORS = {"red": (0, 11), "blue": (2, 22)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def design(seed):
    """One matched scene group. Frames and four queries are nested diagnostics."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    order = list(LAYOUTS)
    random.Random(seed).shuffle(order)
    return {"schema": SCHEMA, "seed": seed, "private_run_order": order,
            "scene_groups": 1, "independent_replicates_per_layout": 1,
            "intervention": "swap target positions only; retain identities, wall and camera script",
            "question": "target left/right of the other object in last mutually visible image",
            "history_selection": "last accepted seen joint frame; current=last accepted hidden joint frame",
            "public_timing": "ordered snapshots only; elapsed simulation times stay in private manifest",
            "acceptance": ["matched authored scenes except target positions",
                           "both sequences pass visibility/timeline/pose audit",
                           "all hidden decoded RGB frames identical within/across layouts at native and 256",
                           "seen RGB-only color ordering agrees with private instance ordering in every frame",
                           "identical current-only requests and opposite private answers for every query"],
            "retry_policy": "one capture per scheduled layout; preserve failures; no automatic retries",
            "model_inference": False, "training": False, "flight": False}


def verify_worlds(trees):
    if set(trees) != set(LAYOUTS):
        raise ValueError("Both layouts required")
    normalized = []
    for layout in LAYOUTS:
        tree = deepcopy(trees[layout])
        for name in ("probe_red", "probe_blue"):
            pose = tree.getroot().find(f"world/model[@name='{name}']/pose")
            x, y, z, yaw = target_poses(layout)[name]
            actual = tuple(float(v) for v in pose.text.split())
            if actual != (x, y, z, 0., 0., yaw):
                raise ValueError("Wrong authored target intervention")
            pose.text = "MATCHED_TARGET_POSE"
        normalized.append(ET.tostring(tree.getroot()))
    if normalized[0] != normalized[1]:
        raise ValueError("Paired scenes changed content beyond target positions")


def rgb_order(rgb):
    """Transparent RGB-only availability reference, not a learned memory model."""
    rgb = rgb.astype(np.int16)
    centers = {}
    for color, (channel, _) in COLORS.items():
        others = [i for i in range(3) if i != channel]
        mask = ((rgb[:, :, channel] > rgb[:, :, others[0]] + 40)
                & (rgb[:, :, channel] > rgb[:, :, others[1]] + 40))
        if int(mask.sum()) < 16:
            raise ValueError("RGB reference needs both colored objects")
        centers[color] = float(np.nonzero(mask)[1].mean())
    return orders(centers)


def orders(centers):
    if (set(centers) != set(COLORS) or not all(np.isfinite(v) for v in centers.values())
            or abs(centers["red"] - centers["blue"]) < 4):
        raise ValueError("Object ordering not separated by four pixels")
    red = "left" if centers["red"] < centers["blue"] else "right"
    return {"red": red, "blue": "right" if red == "left" else "left"}


def label_order(labels):
    centers = {}
    for color, (_, label) in COLORS.items():
        mask = labels == label
        if int(mask.sum()) < 16:
            raise ValueError("Private ordering needs both visible objects")
        centers[color] = float(np.nonzero(mask)[1].mean())
    return orders(centers)


def equal_hidden(arrays):
    """Exact decoded pixels, never similarity thresholds or replacement images."""
    if len(arrays) < 2 or any(not np.array_equal(arrays[0], a) for a in arrays[1:]):
        raise ValueError("Hidden RGB differs; current-image shortcut not excluded")
    return hashlib.sha256(arrays[0].tobytes()).hexdigest()


def png64(rgb):
    stream = BytesIO()
    PILImage.fromarray(rgb).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def request(agent, color, current, history=None):
    if agent not in AGENTS or color not in COLORS or set(current) != set(AGENTS):
        raise ValueError("Invalid diagnostic query")
    other = "blue" if color == "red" else "red"
    payload = {"schema": SCHEMA, "question":
               f"In {agent}'s last image where both objects were visible, was the {color} object "
               f"to the left or right of the {other} object? Answer only left or right.",
               "current_rgb_png_base64": current}
    if history is not None:
        if set(history) != set(AGENTS):
            raise ValueError("Both cameras required in historical snapshot")
        payload["past_rgb_snapshots_png_base64"] = [history]
    return payload


def verify_queries(rows):
    if len(rows) != 2:
        raise ValueError("Exactly two paired query records required")
    a, b = rows
    if canonical(a["current_only"]) != canonical(b["current_only"]):
        raise ValueError("Current-only public requests differ")
    if canonical(a["with_history"]) == canonical(b["with_history"]):
        raise ValueError("Historical requests must differ")
    for row in rows:
        historical = deepcopy(row["with_history"])
        history = historical.pop("past_rgb_snapshots_png_base64", None)
        if not history or canonical(historical) != canonical(row["current_only"]):
            raise ValueError("Only historical RGB may distinguish the public requests")
    if {r["answer"] for r in rows} != {"left", "right"}:
        raise ValueError("Paired private answers must be opposite")


def load_pair(folder, record, spec):
    from gz.msgs10.image_pb2 import Image
    return pair_rgb_instance(rgb_message=_load_message(folder, record["rgb"], Image),
                             label_message=_load_message(folder, record["labels"], Image),
                             rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                             label_topic=spec["segmentation"] + "/labels_map",
                             encoding=GZ8_INSTANCE_ENCODING)


def inspect(output):
    output = Path(output)
    plan = json.loads((output / "design.json").read_text(encoding="utf-8"))
    if plan != design(plan["seed"]):
        raise ValueError("Design does not match the frozen diagnostic protocol")
    summaries, views, answers, trees = {}, {}, {}, {}
    hidden = {agent: {"native": [], "policy_256": []} for agent in AGENTS}
    sheet = PILImage.new("RGB", (1024, 2 * 282), "white")
    for index, layout in enumerate(plan["private_run_order"]):
        folder = output / f"episode_{index}"
        inspect_sequence(folder)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary.get("private_layout") != layout:
            raise ValueError("Capture order/layout differs from frozen plan")
        summaries[layout] = summary
        trees[layout] = ET.parse(folder / "world.sdf")
        specs = {s["agent_id"]: s for s in summary["camera_pairs"]}
        views[layout], answers[layout] = {}, {}
        for phase in summary["phases"][:2]:
            phase_views, phase_answers = {}, {}
            for frame in phase["frames"]:
                for agent in AGENTS:
                    pair = load_pair(folder, frame["agents"][agent], specs[agent])
                    small = resize_pair(pair)
                    if phase["phase"] == "hidden":
                        hidden[agent]["native"].append(pair.rgb)
                        hidden[agent]["policy_256"].append(small.rgb)
                    else:
                        expected = label_order(small.object_labels)
                        if rgb_order(small.rgb) != expected:
                            raise ValueError("RGB reference disagrees with private ordering")
                        if agent in phase_answers and phase_answers[agent] != expected:
                            raise ValueError("Seen ordering changed within a phase")
                        phase_answers[agent] = expected
                    # Deterministically select the last accepted frame, not a successful candidate.
                    phase_views[agent] = png64(small.rgb)
                    if frame is phase["frames"][-1]:
                        col = (0 if phase["phase"] == "seen" else 2) + int(agent[-1])
                        sheet.paste(PILImage.fromarray(small.rgb), (col * 256, index * 282 + 26))
                        ImageDraw.Draw(sheet).text((col * 256 + 4, index * 282 + 5),
                                                  f"episode {index} / {phase['phase']} / {agent}", fill="black")
            views[layout][phase["phase"]] = phase_views
            if phase["phase"] == "seen":
                answers[layout] = phase_answers
    verify_worlds(trees)
    if (len({s["partition"] for s in summaries.values()}) != 2
            or summaries["normal"]["model_resources"] != summaries["swapped"]["model_resources"]):
        raise ValueError("Independent partitions and identical resource manifests required")
    hashes = {agent: {size: equal_hidden(arrays) for size, arrays in sizes.items()}
              for agent, sizes in hidden.items()}
    queries = {}
    for agent in AGENTS:
        for color in COLORS:
            key = f"{agent}_{color}"
            rows = []
            for layout in plan["private_run_order"]:
                rows.append({"current_only": request(agent, color, views[layout]["hidden"]),
                             "with_history": request(agent, color, views[layout]["hidden"], views[layout]["seen"]),
                             "answer": answers[layout][agent][color]})
            verify_queries(rows)
            queries[key] = {"current_only_sha256": hashlib.sha256(canonical(rows[0]["current_only"])).hexdigest(),
                            "private_answers_in_run_order": [r["answer"] for r in rows],
                            "with_history_sha256": [hashlib.sha256(canonical(r["with_history"])).hexdigest()
                                                    for r in rows]}
            for index, row in enumerate(rows):
                # Exports contain only RGB + question. Answers/layout/order stay in private inspection.
                for mode in ("current_only", "with_history"):
                    (output / f"episode_{index}" / f"{key}_{mode}.json").write_bytes(canonical(row[mode]))
    result = {"schema": SCHEMA, "history_information_pair_verified": True,
              "scene_groups": 1, "queries_per_layout": 4, "hidden_rgb_sha256": hashes,
              "queries": queries, "rgb_reference_agrees_with_private_labels": True,
              "current_only_deterministic_pair_accuracy_ceiling": .5,
              "public_protocol": "ordered RGB snapshots; no timestamps, state, layout or answers",
              "not_standard_model_observation": True,
              "memory_use_verified": False, "model_inference": False,
              "world_model_prediction_verified": False, "teammate_adaptation_verified": False,
              "flight_verified": False, "compositional_transfer_verified": False}
    sheet.save(output / "history_pair.png")
    (output / "inspection.json").write_bytes(canonical(result))
    return result


def run(models, output, seed=20261018, timeout_s=90.):
    plan = design(seed)
    verify_worlds({layout: sequence_world(models, layout)[0] for layout in LAYOUTS})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Freeze order and rejection rules before starting any simulator.
    (output / "design.json").write_bytes(canonical(plan))
    for index, layout in enumerate(plan["private_run_order"]):
        print(f"capture episode={index} private_layout={layout}", flush=True)
        summary = capture(models, output / f"episode_{index}", timeout_s, layout=layout)
        if (not summary.get("capture_complete") or summary.get("error")
                or summary.get("cleanup_errors") or not summary.get("owned_process_stopped")):
            raise RuntimeError("Scheduled capture failed; preserve summary; no automatic retry")
    return inspect(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--timeout-s", type=float, default=90.)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        result = inspect(args.output_dir)
    else:
        if args.models is None:
            parser.error("--models required for capture")
        result = run(args.models, args.output_dir, args.seed, args.timeout_s)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
