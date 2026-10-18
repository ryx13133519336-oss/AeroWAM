"""Offline private inspection of no-arm captures. Never constructs policy state.

Only actual RGB JPEGs are exported to a public_images directory. Private
pixel IDs/bboxes/diagnostics stay separate. Saved telemetry is latest receive
state, not capture-aligned, so this does not fabricate ModelObservation data.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .fixtures import Appearance
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .images import ros_image_to_jpeg
from .language_tasks import templated_task
from .pixel_annotations import CaptureKey, annotate


def inspect(directory: Path):
    from gz.msgs10.image_pb2 import Image
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("schema") != "aerowam-annotation-startup/1-draft" or not summary.get("startup_passed")
            or summary.get("error") or summary.get("cleanup_errors") or not summary.get("owned_processes_stopped")):
        raise ValueError("Require a successful stopped startup probe")
    if hashlib.sha256((directory / "world.sdf").read_bytes()).hexdigest() != summary["world_sha256"]:
        raise ValueError("Saved world fingerprint mismatch")
    public = directory / "public_images"
    private = directory / "private_annotations"
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    objects = {"probe_red": Appearance("red", "sphere", "black"),
               "probe_blue": Appearance("blue", "cube", "none")}
    task = templated_task("search", objects["probe_red"]).bind(objects)
    result = {"schema": "aerowam-annotation-startup-inspection/1-draft", "agents": {},
              "capture_aligned_telemetry": False, "policy_delivery_verified": False,
              "flight_ready": False, "live_multitask_ready": False}
    for specification in summary["camera_pairs"]:
        agent = specification["agent_id"]
        if agent not in ("uav_0", "uav_1") or agent in result["agents"]:
            raise ValueError("Invalid/duplicate camera identity")
        saved = summary["saved_captures"][agent]
        messages = []
        for kind in ("rgb", "labels"):
            item = saved["files"][kind]
            if Path(item["file"]).name != item["file"]:
                raise ValueError("Capture path must stay inside diagnostic")
            payload = (directory / item["file"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("Capture fingerprint mismatch")
            message = Image()
            message.ParseFromString(payload)
            messages.append(message)
        pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                                 rgb_topic=specification["rgb"], segmentation_topic=specification["segmentation"],
                                 label_topic=specification["segmentation"] + "/labels_map", encoding=GZ8_INSTANCE_ENCODING)
        if pair.timestamp_ns != saved["stamp_ns"]:
            raise ValueError("Raw capture/manifest stamp mismatch")
        transformed = resize_pair(pair)
        # This diagnostic's static scene explicitly labels all target objects;
        # unlabelled terrain/aircraft are scenery. Unknown labels still reject.
        key = CaptureKey(agent, pair.timestamp_ns, pair.timestamp_ns,
                         f"startup-static-{summary['partition']}-{agent}-{pair.timestamp_ns}", 256, 256)
        bundle = annotate(task=task, rgb_capture=key, label_capture=key,
                          instance_labels=transformed.object_labels,
                          label_to_object={0: None, 11: "probe_red", 22: "probe_blue", 33: None})
        rgb = messages[0]
        ros_like = SimpleNamespace(width=rgb.width, height=rgb.height, encoding="rgb8", step=rgb.step, data=rgb.data)
        jpeg = ros_image_to_jpeg(ros_like)
        (public / f"{agent}.jpg").write_bytes(jpeg)
        PILImage.fromarray(transformed.object_labels).save(private / f"{agent}_labels.png")
        overlay = PILImage.fromarray(transformed.rgb)
        draw = ImageDraw.Draw(overlay)
        for annotation in bundle.frame.objects.values():
            if annotation.visible_bbox is not None:
                x1, y1, x2, y2 = annotation.visible_bbox
                draw.rectangle((round(x1*256), round(y1*256), round(x2*256)-1, round(y2*256)-1), outline="yellow")
        overlay.save(private / f"{agent}_overlay.png")
        result["agents"][agent] = {"image_timestamp_ns": pair.timestamp_ns,
                                   "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                                   "native_size": [rgb.width, rgb.height],
                                   "label_values": np.unique(pair.object_labels).tolist(),
                                   "annotations": bundle.diagnostics}
    if set(result["agents"]) != {"uav_0", "uav_1"}:
        raise ValueError("Both cameras required")
    (private / "inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    print(json.dumps(inspect(parser.parse_args().probe_dir), indent=2))


if __name__ == "__main__":
    main()
