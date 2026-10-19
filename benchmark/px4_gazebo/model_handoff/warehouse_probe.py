"""External warehouse static rendering/annotation admission. Never arms PX4."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .annotation_probe import run as capture
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .pixel_annotations import CaptureKey, annotate
from .search_pilot_scene import LABELS, manual_condition
from .warehouse_scene import CASES, WORLD, static_world, prepare, sha256


def inspect(directory):
    from gz.msgs10.image_pb2 import Image
    directory = Path(directory)
    result = {"schema": "aerowam-warehouse-static-probe/1-draft", "cases": {},
              "world_sha256": {},
              "static_admission_pass": False, "flight_verified": False,
              "policy_evaluated": False, "compositional_transfer_verified": False}
    sheet = PILImage.new("RGB", (1024, len(CASES) * 414), "white")
    normalized = []
    resources = []
    for row, case in enumerate(CASES):
        folder = directory / case
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("error") or summary.get("cleanup_errors")
                or not summary.get("owned_process_stopped")):
            raise ValueError(f"{case}: capture/cleanup failed")
        if sha256(folder / "world.sdf") != summary["world_sha256"]:
            raise ValueError("World fingerprint mismatch")
        result["world_sha256"][case] = summary["world_sha256"]
        tree = ET.parse(folder / "world.sdf")
        world = tree.getroot().find("world")
        if world.get("name") != WORLD:
            raise ValueError("Unexpected warehouse world")
        specs = summary["camera_pairs"]
        if len(specs) != 2 or {s["agent_id"] for s in specs} != {"uav_0", "uav_1"}:
            raise ValueError("Two distinct camera pairs required")
        for i in range(2):
            camera = world.find(f"model[@name='annotation_camera_{i}']")
            camera.remove(camera.find("pose"))
        normalized.append(ET.tostring(tree.getroot()))
        resources.append({key: summary["scene_resources"][key] for key in ("warehouse", "cameras")})
        findings = {}
        for spec in specs:
            agent = spec["agent_id"]
            prefix = f"/world/{WORLD}/model/annotation_camera_{agent[-1]}/link/camera_link/sensor"
            if spec["rgb"] != prefix + "/camera/image" or spec["segmentation"] != prefix + "/private_instance/segmentation":
                raise ValueError("Unexpected camera topic binding")
            rgb_rows = {m["stamp_ns"]: m for m in summary["messages"].get(spec["rgb"], [])}
            label_rows = {m["stamp_ns"]: m for m in summary["messages"].get(spec["segmentation"] + "/labels_map", [])}
            if (len(rgb_rows) != len(summary["messages"].get(spec["rgb"], []))
                    or len(label_rows) != len(summary["messages"].get(spec["segmentation"] + "/labels_map", []))):
                raise ValueError("Duplicate capture timestamp")
            common = sorted(rgb_rows.keys() & label_rows.keys())
            if len(common) < 3:
                raise ValueError("Three distinct exact-stamp pairs required per view")
            findings[agent] = []
            for stamp in common:
                messages, hashes = [], []
                for record in (rgb_rows[stamp], label_rows[stamp]):
                    filename = record["file"]
                    if Path(filename).name != filename:
                        raise ValueError("Raw message must remain inside capture directory")
                    raw = (folder / filename).read_bytes()
                    message = Image()
                    message.ParseFromString(raw)
                    messages.append(message)
                    hashes.append(hashlib.sha256(raw).hexdigest())
                pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                                         rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                                         label_topic=spec["segmentation"] + "/labels_map",
                                         encoding=GZ8_INSTANCE_ENCODING)
                if pair.timestamp_ns != stamp:
                    raise ValueError("Capture timestamp mismatch")
                frames = {"stamp_ns": stamp, "rgb_sha256": hashes[0], "labels_sha256": hashes[1]}
                for resolution, raster in (("native", pair), ("policy_256", resize_pair(pair))):
                    height, width = raster.object_labels.shape
                    key = CaptureKey(agent, stamp, stamp, summary["partition"], width, height)
                    annotation = annotate(task=manual_condition("寻找红球并确认", "red", role_mode="unassigned"),
                                          rgb_capture=key, label_capture=key,
                                          instance_labels=raster.object_labels,
                                          label_to_object={**LABELS, 100: None})
                    pixels = {identity: int(np.count_nonzero(raster.object_labels == label))
                              for label, identity in LABELS.items() if identity is not None}
                    if np.count_nonzero(raster.object_labels == 100) < 16:
                        raise ValueError("Rendered warehouse background support missing")
                    # Check physical RGB support only for visible targets. No
                    # color-threshold detector substitutes for private labels.
                    for label, identity in LABELS.items():
                        if identity is None or pixels[identity] < 16:
                            continue
                        rgb = raster.rgb.astype(np.int16)
                        channel = 0 if label == 11 else 2
                        other = [c for c in range(3) if c != channel]
                        colored = ((raster.object_labels == label)
                                   & (rgb[:, :, channel] > rgb[:, :, other[0]] + 30)
                                   & (rgb[:, :, channel] > rgb[:, :, other[1]] + 30))
                        if int(colored.sum()) < 8:
                            raise ValueError("Rendered RGB and instance labels disagree")
                    frames[resolution] = {"size": [width, height], "target_pixels": pixels,
                                           "background_pixels": int(np.count_nonzero(raster.object_labels == 100)),
                                           "visible_boxes": {name: obj.visible_bbox for name, obj in annotation.frame.objects.items()}}
                findings[agent].append(frames)
                if stamp == common[0]:
                    preview = PILImage.fromarray(pair.rgb).resize((512, 384))
                    preview.save(folder / f"{agent}_rgb.png")
                    overlay = preview.copy()
                    draw = ImageDraw.Draw(overlay)
                    for name, box in frames["native"]["visible_boxes"].items():
                        if box:
                            coordinates = tuple(v * (512 if i % 2 == 0 else 384) for i, v in enumerate(box))
                            draw.rectangle(coordinates, outline="yellow", width=2)
                            draw.text(coordinates[:2], name, fill="yellow")
                    overlay.save(folder / f"{agent}_private_overlay.png")
                    col = int(agent[-1]) * 512
                    sheet.paste(overlay, (col, row * 414 + 30))
                    ImageDraw.Draw(sheet).text((col + 5, row * 414 + 8), f"{case} / {agent} (private QA)", fill="black")
        result["cases"][case] = findings
    if len(set(normalized)) != 1 or any(resource != resources[0] for resource in resources):
        raise ValueError("Viewpoint controls must retain identical assets and targets")
    # Admission requires a physical reveal caused only by camera pose changes.
    # These are independent static captures, not memory or moving-UAV evidence.
    revealed = []
    for identity in ("probe_red", "probe_blue"):
        hidden = any(all(f["policy_256"]["target_pixels"][identity] == 0 for f in frames)
                     for frames in result["cases"]["rack_front"].values())
        visible = any(all(f["policy_256"]["visible_boxes"][identity] is not None for f in frames)
                      for frames in result["cases"]["rack_back"].values())
        if hidden and visible:
            revealed.append(identity)
    result["revealed_targets"] = revealed
    result["matched_viewpoint_controls"] = True
    result["static_admission_pass"] = len(revealed) == 2
    sheet.save(directory / "warehouse_views.png")
    (directory / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Pinned upstream ZIP; prepare a new local package")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.)
    args = parser.parse_args()
    if not args.inspect_only:
        if args.models is None:
            parser.error("--models required for paired native-resolution sensors")
        if args.archive:
            prepare(args.archive, args.package)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for case in CASES:
            prepared = static_world(args.package, args.models, case)
            summary = capture(args.output_dir / case, args.timeout_s, case, prepared_scene=prepared)
            if summary.get("error") or summary.get("cleanup_errors") or not summary.get("owned_process_stopped"):
                raise RuntimeError(f"{case}: capture failed; inspect summary.json")
    result = inspect(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    raise SystemExit(0 if result["static_admission_pass"] else 2)


if __name__ == "__main__":
    main()
