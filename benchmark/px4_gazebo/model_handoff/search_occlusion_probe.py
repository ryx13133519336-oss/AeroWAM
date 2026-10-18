"""Private static search occlusion controls. No PX4, policy, flight or score."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .annotation_probe import run as capture
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .pixel_annotations import CaptureKey, annotate
from .search_pilot_scene import (
    LABELS, OBJECTS, OCCLUSION_CASES, OCCLUSION_WORLD, manual_condition,
)

SCHEMA = "aerowam-search-occlusion-probe/1-draft"


def verify_controls(trees):
    """Reject changed targets, intrinsics, wall, or unintended scene changes."""
    if set(trees) != set(OCCLUSION_CASES):
        raise ValueError("All three matched static controls are required")
    normalized, poses, walls = {}, {}, {}
    for case, original in trees.items():
        tree = deepcopy(original)
        world = tree.getroot().find("world")
        if world is None or world.get("name") != OCCLUSION_WORLD:
            raise ValueError("Unexpected diagnostic world")
        expected = {*OBJECTS, "annotation_camera_0", "annotation_camera_1"}
        if case != "search_clear":
            expected.add("probe_wall")
        models = world.findall("model")
        if (len(models) != len(expected) or {model.get("name") for model in models} != expected
                or world.findall("include")):
            raise ValueError("Unexpected scene entities")
        if any(model.findtext("static") != "true" for model in world.findall("model")):
            raise ValueError("Diagnostic entities must be static")
        wall = world.find("model[@name='probe_wall']")
        if wall is not None:
            walls[case] = ET.tostring(wall)
            world.remove(wall)
        poses[case] = []
        for index in range(2):
            camera = world.find(f"model[@name='annotation_camera_{index}']")
            pose = camera.find("pose")
            values = tuple(float(value) for value in pose.text.split())
            if len(values) != 6 or not all(np.isfinite(values)):
                raise ValueError("Invalid static camera pose")
            poses[case].append(values)
            camera.remove(pose)
        normalized[case] = ET.tostring(tree.getroot())
    if len(set(normalized.values())) != 1:
        raise ValueError("Controls changed assets, camera resources or environment")
    if poses["search_clear"] != poses["search_blocked"]:
        raise ValueError("Clear and blocked cameras must match")
    if any(a == b for a, b in zip(poses["search_blocked"], poses["search_revealed"])):
        raise ValueError("Both revealed viewpoints must change")
    if walls["search_blocked"] != walls["search_revealed"]:
        raise ValueError("Revealed control must retain the exact wall")
    return {"matched_scene_controls": True, "camera_poses_enu": poses}


def measure(pair, *, scene_case, agent_id, render_id):
    """Private instance pixels determine visibility; RGB is a sanity check."""
    if scene_case not in OCCLUSION_CASES:
        raise ValueError("Unknown occlusion case")
    labels = {**LABELS, 33: None}
    records = {}
    for resolution, raster in (("native", pair), ("policy_256", resize_pair(pair))):
        height, width = raster.object_labels.shape
        key = CaptureKey(agent_id, raster.timestamp_ns, raster.timestamp_ns,
                         render_id, width, height)
        bundle = annotate(task=manual_condition("寻找红球并确认", "red"),
                          rgb_capture=key, label_capture=key,
                          instance_labels=raster.object_labels, label_to_object=labels)
        counts = {identity: int(np.count_nonzero(raster.object_labels == label))
                  for label, identity in LABELS.items() if identity is not None}
        blocked = scene_case == "search_blocked"
        for label, identity in LABELS.items():
            if identity is None:
                continue
            box = bundle.frame.objects[identity].visible_bbox
            if blocked:
                if counts[identity] != 0 or box is not None:
                    raise ValueError(f"{agent_id}/{resolution}: target not fully occluded: {identity}")
            else:
                if box is None:
                    raise ValueError(f"{agent_id}/{resolution}: target not reportable: {identity}")
                rgb = raster.rgb.astype(np.int16)
                channel = 0 if label == 11 else 2
                other = [i for i in range(3) if i != channel]
                colored = ((raster.object_labels == label)
                           & (rgb[:, :, channel] > rgb[:, :, other[0]] + 40)
                           & (rgb[:, :, channel] > rgb[:, :, other[1]] + 40))
                if int(colored.sum()) < 16:
                    raise ValueError("Target RGB/label appearance sanity check failed")
        wall_pixels = int(np.count_nonzero(raster.object_labels == 33))
        if blocked and wall_pixels < 16:
            raise ValueError("Blocked view needs rendered wall support")
        if scene_case == "search_clear" and wall_pixels:
            raise ValueError("Clear control unexpectedly contains wall pixels")
        records[resolution] = {"size": [width, height], "target_pixels": counts,
                               "wall_pixels": wall_pixels,
                               "visible_boxes": {identity: item.visible_bbox
                                                 for identity, item in bundle.frame.objects.items()}}
    return records


def inspect(directory):
    """Inspect three independent captures; never label them a time sequence."""
    from gz.msgs10.image_pb2 import Image
    directory = Path(directory)
    summaries, trees = {}, {}
    for case in OCCLUSION_CASES:
        folder = directory / case
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("scene_case") != case or summary.get("error")
                or summary.get("cleanup_errors") or not summary.get("owned_process_stopped")):
            raise ValueError("Capture or cleanup failed; inspect case summary")
        raw = (folder / "world.sdf").read_bytes()
        if hashlib.sha256(raw).hexdigest() != summary.get("world_sha256"):
            raise ValueError("Scene fingerprint mismatch")
        trees[case] = ET.ElementTree(ET.fromstring(raw))
        summaries[case] = summary
    controls = verify_controls(trees)
    resources = summaries["search_clear"].get("model_resources", {})
    if resources.get("schema") != "aerowam-annotation-models/1-draft" or not resources.get("files"):
        raise ValueError("Paired camera resource fingerprints required")
    if any(summaries[c].get("model_resources") != resources
           for c in OCCLUSION_CASES):
        raise ValueError("Camera resource manifests changed between controls")
    result = {"schema": SCHEMA, **controls, "cases": {},
              "world_sha256": {case: summaries[case]["world_sha256"] for case in OCCLUSION_CASES},
              "static_visibility_controls_pass": False,
              "temporal_reacquisition_verified": False, "memory_use_verified": False,
              "flight_verified": False, "compositional_transfer_verified": False}
    contact_sheet = PILImage.new("RGB", (512, 3 * 282), "white")
    for row_index, case in enumerate(OCCLUSION_CASES):
        summary = summaries[case]
        specs = summary.get("camera_pairs", [])
        if len(specs) != 2 or {s["agent_id"] for s in specs} != {"uav_0", "uav_1"}:
            raise ValueError("Exactly two distinct camera pairs required")
        findings = {}
        for spec in specs:
            agent = spec["agent_id"]
            expected = f"/world/{OCCLUSION_WORLD}/model/annotation_camera_{agent[-1]}/link/camera_link/sensor"
            if (spec["rgb"] != expected + "/camera/image"
                    or spec["segmentation"] != expected + "/private_instance/segmentation"):
                raise ValueError("Unexpected camera topic mapping")
            def indexed(topic):
                rows = summary["messages"].get(topic, [])
                mapping = {item["stamp_ns"]: item for item in rows}
                if len(mapping) != len(rows):
                    raise ValueError("Duplicate capture timestamps")
                return mapping
            rgb_rows = indexed(spec["rgb"])
            label_rows = indexed(spec["segmentation"] + "/labels_map")
            common = sorted(set(rgb_rows) & set(label_rows))
            if len(common) < 3:
                raise ValueError("At least three fresh exact-stamp pairs per camera required")
            frames = []
            for stamp in common:
                messages = []
                hashes = []
                for metadata in (rgb_rows[stamp], label_rows[stamp]):
                    name = metadata["file"]
                    if Path(name).name != name:
                        raise ValueError("Raw message path must stay in case directory")
                    raw_message = (directory / case / name).read_bytes()
                    hashes.append(hashlib.sha256(raw_message).hexdigest())
                    message = Image()
                    message.ParseFromString(raw_message)
                    messages.append(message)
                pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                                         rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                                         label_topic=spec["segmentation"] + "/labels_map",
                                         encoding=GZ8_INSTANCE_ENCODING)
                if pair.timestamp_ns != stamp:
                    raise ValueError("Raw capture stamp differs from manifest")
                metrics = measure(pair, scene_case=case, agent_id=agent,
                                  render_id=f"static-{summary['partition']}-{agent}-{stamp}")
                frames.append({"stamp_ns": stamp, "rgb_sha256": hashes[0],
                               "labels_sha256": hashes[1], **metrics})
                if len(frames) == 1:
                    preview = PILImage.fromarray(resize_pair(pair).rgb)
                    draw = ImageDraw.Draw(preview)
                    for identity, box in metrics["policy_256"]["visible_boxes"].items():
                        if box:
                            draw.rectangle(tuple(int(v * 256) for v in box), outline="yellow", width=2)
                            draw.text((int(box[0] * 256), int(box[1] * 256)), identity, fill="yellow")
                    preview.save(directory / case / f"{agent}_overlay.png")
                    col = int(agent[-1]) * 256
                    contact_sheet.paste(preview, (col, row_index * 282 + 26))
                    ImageDraw.Draw(contact_sheet).text((col + 4, row_index * 282 + 5), f"{case} / {agent}", fill="black")
            findings[agent] = frames
        result["cases"][case] = findings
    result["static_visibility_controls_pass"] = True
    contact_sheet.save(directory / "visibility_controls.png")
    (directory / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=40.)
    args = parser.parse_args()
    if not args.inspect_only:
        if args.models is None:
            parser.error("--models required for native paired-camera capture")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for case in OCCLUSION_CASES:
            summary = capture(args.output_dir / case, args.timeout_s, case, args.models)
            if summary.get("error") or summary.get("cleanup_errors") or not summary.get("owned_process_stopped"):
                raise RuntimeError(f"{case}: capture failed; inspect summary.json")
    result = inspect(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
