"""Offline inspection of the isolated static probe, never a flight score."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw

from .fixtures import Appearance
from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance, resize_pair
from .language_tasks import templated_task
from .pixel_annotations import CaptureKey, annotate

RGB = "/aerowam/annotation_probe/rgb"
SEG = "/aerowam/annotation_probe/segmentation"
LABEL = SEG + "/labels_map"


def inspect(directory: Path, agent_id: str = "uav_0"):
    from gz.msgs10 import image_pb2
    # Confirm the installed wire enum; avoid interpreting BGR/float as labels.
    if image_pb2.DESCRIPTOR.enum_types_by_name["PixelFormatType"].values_by_name["RGB_INT8"].number != 3:
        raise RuntimeError("Installed Gazebo wire enum differs from decoder contract")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("error") or summary.get("cleanup_errors") or not summary.get("owned_process_stopped"):
        raise RuntimeError("Probe capture/cleanup failed; inspect summary.json")
    if "world_sha256" in summary and hashlib.sha256((directory / "world.sdf").read_bytes()).hexdigest() != summary["world_sha256"]:
        raise ValueError("Saved diagnostic scene fingerprint mismatch")
    scene_case = summary.get("scene_case", "partial")
    if scene_case not in {"partial", "clear", "full"}:
        raise ValueError("Unknown static probe scene case")
    streams = summary["messages"]
    specifications = summary.get("camera_pairs", [{"agent_id": "uav_0", "rgb": RGB, "segmentation": SEG}])
    selected = [item for item in specifications if item["agent_id"] == agent_id]
    if len(selected) != 1:
        raise ValueError("Missing or ambiguous camera identity")
    rgb_topic, seg_topic = selected[0]["rgb"], selected[0]["segmentation"]
    prefix = agent_id + "_" if len(specifications) > 1 else ""
    rgb_rows = {row["stamp_ns"]: row for row in streams.get(rgb_topic, [])}
    label_rows = {row["stamp_ns"]: row for row in streams.get(seg_topic + "/labels_map", [])}
    common = sorted(set(rgb_rows) & set(label_rows))
    if len(common) < 3:
        raise RuntimeError("Fewer than three exact-stamp RGB/label pairs")
    objects = {"probe_red": Appearance("red", "sphere", "black"),
               "probe_blue": Appearance("blue", "cube", "none")}
    task = templated_task("search", objects["probe_red"]).bind(objects)
    findings = []
    for index, stamp in enumerate(common):
        messages = []
        for row in (rgb_rows[stamp], label_rows[stamp]):
            filename = row["file"]
            if Path(filename).name != filename:
                raise ValueError("Raw message path must stay in probe directory")
            message = image_pb2.Image()
            message.ParseFromString((directory / filename).read_bytes())
            messages.append(message)
        pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                                 rgb_topic=rgb_topic, segmentation_topic=seg_topic, label_topic=seg_topic + "/labels_map",
                                 encoding=GZ8_INSTANCE_ENCODING)
        if pair.timestamp_ns != stamp:
            raise ValueError("Raw message/manifest stamp mismatch")
        values, counts = np.unique(pair.object_labels, return_counts=True)
        channel_values = [np.unique(messages_array).tolist() for messages_array in
                          (pair.instance_bytes[:, :, 0], pair.instance_bytes[:, :, 1], pair.object_labels)]
        if index == 0:
            PILImage.fromarray(pair.rgb).save(directory / f"{prefix}rgb.png")
            np.save(directory / f"{prefix}object_labels.npy", pair.object_labels, allow_pickle=False)
            PILImage.fromarray(pair.object_labels).save(directory / f"{prefix}object_labels.png")
        # This key identifies a static, co-configured pair, not a claim of
        # capture-aligned moving drone/object geometry or policy delivery.
        key = CaptureKey(agent_id, stamp, stamp, f"static-pair-{summary['partition']}-{agent_id}-{stamp}",
                         pair.rgb.shape[1], pair.rgb.shape[0])
        bundle = annotate(task=task, rgb_capture=key, label_capture=key,
                          instance_labels=pair.object_labels,
                          label_to_object={0: None, 11: "probe_red", 22: "probe_blue", 33: None})
        expected = {"partial": {0, 11, 22, 33}, "clear": {0, 11, 22}, "full": {0, 22, 33}}[scene_case]
        if set(values.tolist()) != expected:
            raise ValueError(f"Static scene labels missing/unexpected: {values.tolist()}")
        if bundle.frame.objects["probe_blue"].visible_bbox is None:
            raise ValueError("Static visible object rejected by pixel annotation rules")
        if (bundle.frame.objects["probe_red"].visible_bbox is None) != (scene_case == "full"):
            raise ValueError("Red target visibility disagrees with static occlusion condition")
        # Diagnostic RGB cross-check only; color does not produce truth labels.
        rgb16 = pair.rgb.astype(np.int16)
        red = pair.object_labels == 11
        blue = pair.object_labels == 22
        red_body = red & (rgb16[:, :, 0] > rgb16[:, :, 1] + 40) & (rgb16[:, :, 0] > rgb16[:, :, 2] + 40)
        black_stripe = red & (np.max(rgb16, axis=2) < 50)
        blue_body = blue & (rgb16[:, :, 2] > rgb16[:, :, 0] + 40) & (rgb16[:, :, 2] > rgb16[:, :, 1] + 40)
        if int(blue_body.sum()) < 16 or (scene_case != "full" and
                (int(red_body.sum()) < 16 or int(black_stripe.sum()) < 16)):
            raise ValueError("Static appearance/label overlay sanity check failed")
        transformed = resize_pair(pair)
        model_key = CaptureKey(agent_id, stamp, stamp, key.render_id, 256, 256)
        model_bundle = annotate(task=task, rgb_capture=model_key, label_capture=model_key,
                                instance_labels=transformed.object_labels,
                                label_to_object={0: None, 11: "probe_red", 22: "probe_blue", 33: None})
        if (model_bundle.frame.objects["probe_red"].visible_bbox is None) != (scene_case == "full"):
            raise ValueError("Transformed red visibility mismatch")
        if model_bundle.frame.objects["probe_blue"].visible_bbox is None:
            raise ValueError("Transformed blue visibility mismatch")
        if index == 0:
            overlay = PILImage.fromarray(pair.rgb)
            draw = ImageDraw.Draw(overlay)
            w, h = overlay.size
            for identity, annotation in bundle.frame.objects.items():
                if annotation.visible_bbox is None:
                    continue
                x1, y1, x2, y2 = annotation.visible_bbox
                box = (int(x1*w), int(y1*h), int(x2*w)-1, int(y2*h)-1)
                draw.rectangle(box, outline="yellow", width=1)
                draw.text((box[0], max(0, box[1]-12)), identity, fill="yellow")
            overlay.save(directory / f"{prefix}annotation_overlay.png")
            PILImage.fromarray(transformed.rgb).save(directory / f"{prefix}model_rgb.png")
            PILImage.fromarray(transformed.object_labels).save(directory / f"{prefix}model_labels.png")
            model_overlay = PILImage.fromarray(transformed.rgb)
            model_draw = ImageDraw.Draw(model_overlay)
            for annotation in model_bundle.frame.objects.values():
                if annotation.visible_bbox is not None:
                    x1, y1, x2, y2 = annotation.visible_bbox
                    model_draw.rectangle((int(x1*256), int(y1*256), int(x2*256)-1, int(y2*256)-1), outline="yellow")
            model_overlay.save(directory / f"{prefix}model_overlay.png")
        findings.append({"timestamp_ns": stamp, "channel_values": channel_values,
                         "rgb_sanity_pixels": {"red_body": int(red_body.sum()),
                                               "black_stripe": int(black_stripe.sum()),
                                               "blue_body": int(blue_body.sum())},
                         "label_pixels": {str(int(v)): int(c) for v, c in zip(values, counts)},
                         "annotations": bundle.diagnostics, "model_raster_annotations": model_bundle.diagnostics})
    result = {"schema": "aerowam-static-annotation-check/1-draft",
              "scene_case": scene_case,
              "agent_id": agent_id, "native_dimensions": [messages[0].width, messages[0].height],
              "exact_stamp_pairs": len(common), "wire_and_static_annotations_passed": True,
              "moving_capture_calibration_verified": False, "live_multitask_ready": False,
              "frames": findings}
    (directory / f"{prefix}inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.probe_dir / "summary.json").read_text(encoding="utf-8"))
    results = {}
    for specification in summary.get("camera_pairs", [{"agent_id": "uav_0"}]):
        agent = specification["agent_id"]
        result = inspect(args.probe_dir, agent)
        compact = {key: value for key, value in result.items() if key != "frames"}
        compact["first_frame"] = result["frames"][0]
        results[agent] = compact
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
