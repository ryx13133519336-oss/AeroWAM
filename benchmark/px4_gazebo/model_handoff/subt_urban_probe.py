"""Private paired-raster QA for SubT. Static visibility, never model scores."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .pixel_annotations import CaptureKey, annotate
from .search_pilot_scene import LABELS, manual_condition
from .subt_urban_scene import CAMERAS, WORLD, TARGETS, scene, verify
from .warehouse_scene import local_resource, sha256


def inspect_capture(folder):
    from gz.msgs10.image_pb2 import Image
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("error") or summary.get("cleanup_errors") or not summary.get("owned_process_stopped")
            or sha256(folder / "world.sdf") != summary.get("world_sha256")):
        raise ValueError("Capture, cleanup or world fingerprint failed")
    log = (folder / "gazebo.log").read_text(encoding="utf-8", errors="replace")
    if any(error in log for error in ("[Err]", "Unable to load mesh", "Failed to load", "Unable to find")):
        raise ValueError("Gazebo resource error; inspect gazebo.log")
    world = ET.parse(folder / "world.sdf").getroot().find("world")
    if world.get("name") != WORLD:
        raise ValueError("Unexpected SubT world")
    specs = summary["camera_pairs"]
    if len(specs) != 2 or {s["agent_id"] for s in specs} != {"uav_0", "uav_1"}:
        raise ValueError("Two distinct camera bindings required")
    result = {"world_sha256": summary["world_sha256"], "agents": {}, "static_capture_pass": False,
              "flight_verified": False, "policy_evaluated": False}
    for spec in specs:
        agent = spec["agent_id"]
        prefix = f"/world/{WORLD}/model/annotation_camera_{agent[-1]}/link/camera_link/sensor"
        if spec != {"agent_id": agent, "rgb": prefix + "/camera/image",
                    "segmentation": prefix + "/private_instance/segmentation"}:
            raise ValueError("Camera binding mismatch")
        records = []
        for topic in (spec["rgb"], spec["segmentation"] + "/labels_map"):
            rows = summary["messages"].get(topic, [])
            indexed = {r["stamp_ns"]: r for r in rows}
            if len(indexed) != len(rows):
                raise ValueError("Duplicate frame timestamp")
            records.append(indexed)
        stamps = sorted(records[0].keys() & records[1].keys())
        if len(stamps) < 3:
            raise ValueError("Three exact-stamp pairs required")
        result["agents"][agent] = []
        for stamp in stamps:
            messages, hashes = [], []
            for indexed in records:
                raw = local_resource(folder, indexed[stamp]["file"]).read_bytes()
                message = Image()
                message.ParseFromString(raw)
                messages.append(message)
                hashes.append(hashlib.sha256(raw).hexdigest())
            pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                                     rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                                     label_topic=spec["segmentation"] + "/labels_map", encoding=GZ8_INSTANCE_ENCODING)
            if pair.timestamp_ns != stamp:
                raise ValueError("Timestamp metadata mismatch")
            row = {"stamp_ns": stamp, "raw_sha256": hashes}
            for name, raster in (("native", pair), ("policy_256", resize_pair(pair))):
                height, width = raster.object_labels.shape
                capture = CaptureKey(agent, stamp, stamp, summary["partition"], width, height)
                annotation = annotate(task=manual_condition("寻找目标并协作确认", "red", role_mode="unassigned"),
                                      rgb_capture=capture, label_capture=capture,
                                      instance_labels=raster.object_labels, label_to_object={**LABELS, 100: None})
                background = int(np.count_nonzero(raster.object_labels == 100))
                if background < 16:
                    raise ValueError("Rendered building missing from instance raster")
                pixels = {identity: int(np.count_nonzero(raster.object_labels == label))
                          for label, identity in LABELS.items() if identity}
                rgb = raster.rgb.astype(np.int16)
                for label, channel in ((11, 0), (22, 2)):
                    if np.count_nonzero(raster.object_labels == label) < 16:
                        continue
                    other = [c for c in range(3) if c != channel]
                    support = ((raster.object_labels == label) & (rgb[:, :, channel] > rgb[:, :, other[0]] + 30)
                               & (rgb[:, :, channel] > rgb[:, :, other[1]] + 30))
                    if int(support.sum()) < 8:
                        raise ValueError("RGB color support disagrees with labels")
                row[name] = {"size": [width, height], "background_pixels": background, "target_pixels": pixels,
                             "visible_boxes": {key: obj.visible_bbox for key, obj in annotation.frame.objects.items()}}
            result["agents"][agent].append(row)
            if stamp == stamps[0]:
                PILImage.fromarray(pair.rgb).save(folder / f"{agent}_rgb.png")
    result["static_capture_pass"] = True
    (folder / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def inspect_controls(directory, package):
    from .subt_urban_geometry import load_triangles, segment_hits
    directory = Path(directory)
    triangles = load_triangles(package)
    normalized, tables, cases = [], [], {}
    sheet = PILImage.new("RGB", (1024, len(CAMERAS) * 408), "white")
    for index, case in enumerate(CAMERAS):
        folder = directory / case
        cases[case] = inspect_capture(folder)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        contract = summary["scene_resources"]
        if contract["subt"] != verify(package) or contract["case"] != case or contract["role_assignment"] is not None:
            raise ValueError("Unexpected scene provenance or fixed roles")
        tables.append((contract["subt"], contract["cameras"]))
        root = ET.parse(folder / "world.sdf").getroot()
        for identity, xyz in TARGETS.items():
            pose = tuple(map(float, root.findtext(f"world/model[@name='{identity}']/pose").split()))
            if pose != (*xyz, 0., 0., 0.):
                raise ValueError("Unexpected private target pose")
        for camera in root.findall("world/model"):
            if camera.get("name", "").startswith("annotation_camera_"):
                camera_index = int(camera.get("name").rsplit("_", 1)[-1])
                pose = tuple(map(float, camera.findtext("pose").split()))
                if pose != CAMERAS[case][camera_index]:
                    raise ValueError("Unexpected diagnostic camera pose")
                camera.remove(camera.find("pose"))
        normalized.append(ET.tostring(root))
        for col, agent in enumerate(("uav_0", "uav_1")):
            sheet.paste(PILImage.open(folder / f"{agent}_rgb.png").resize((512, 384)), (col * 512, index * 408 + 24))
            ImageDraw.Draw(sheet).text((col * 512 + 8, index * 408 + 6), f"{case} / {agent} - static QA", fill="black")
    if len(set(normalized)) != 1 or any(t != tables[0] for t in tables):
        raise ValueError("View controls altered geometry, targets or sensors")
    rays = {}
    for case in ("hidden", "revealed"):
        rays[case] = {}
        for index, pose in enumerate(CAMERAS[case]):
            agent = f"uav_{index}"
            rays[case][agent] = {}
            for identity, target in TARGETS.items():
                hits = segment_hits(triangles, pose[:3], target)
                rays[case][agent][identity] = len(hits)
                delta = np.asarray(target) - pose[:3]
                angle = np.arctan2(delta[1], delta[0]) - pose[5]
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                if abs(angle) > .6 or (case == "hidden") != bool(len(hits)):
                    raise ValueError("Control must use physical wall occlusion, not FOV exclusion")
                for frame in cases[case]["agents"][agent]:
                    raster = frame["policy_256"]
                    if case == "hidden" and raster["target_pixels"][identity] != 0:
                        raise ValueError("Hidden target still visible")
                    if case == "revealed" and raster["visible_boxes"][identity] is None:
                        raise ValueError("Revealed target lacks a valid box")
    result = {"schema": "aerowam-subt-static-admission/1-draft", "static_admission_pass": True,
              "cases": cases, "matched_viewpoint_controls": True, "collision_triangle_count": len(triangles),
              "center_ray_intersections": rays, "flight_verified": False, "policy_evaluated": False,
              "compositional_transfer_verified": False}
    sheet.save(directory / "subt_urban_views.png")
    (directory / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.)
    args = parser.parse_args()
    if not args.inspect_only:
        if args.models is None:
            parser.error("--models required")
        from .annotation_probe import run
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for case in CAMERAS:
            tree, contract = scene(args.package, args.models, case=case)
            run(args.output_dir / case, args.timeout_s, case,
                prepared_scene=(tree, contract["camera_pairs"], contract))
            inspect_capture(args.output_dir / case)
    result = inspect_controls(args.output_dir, args.package)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
