"""Private moving-scene QA, never policy observations or task success evidence."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import threading

import numpy as np

from .gazebo_rasters import GZ8_INSTANCE_ENCODING, pair_rgb_instance
from .industrial_targets import IDENTITIES, LABELS, color_support, from_configuration


class MotionCapture:
    """Bounded exact RGB/label joins; retain at most two pairs per sim second."""

    def __init__(self, contract):
        self.contract = contract
        self.lock = threading.Lock()
        self.pending = {spec["agent_id"]: [OrderedDict(), OrderedDict()] for spec in contract["camera_pairs"]}
        self.frames = {spec["agent_id"]: [] for spec in contract["camera_pairs"]}
        self.origin_ns = None
        self.errors = []

    def receive(self, agent, channel, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
        with self.lock:
            if self.origin_ns is None or stamp < self.origin_ns:
                return
            rows = self.frames[agent]
            if len(rows) >= 122 or (rows and stamp - rows[-1][0] < 500_000_000):
                return
            buffers = self.pending[agent]
            # Serialized payloads detach callback lifetime and bound pending RAM.
            buffers[channel][stamp] = message.SerializeToString()
            while len(buffers[channel]) > 16:
                buffers[channel].popitem(last=False)
            if stamp not in buffers[1 - channel]:
                return
            rows.append((stamp, buffers[0][stamp], buffers[1][stamp]))
            for buffer in buffers:
                for old in [key for key in buffer if key <= stamp]:
                    del buffer[old]

    def start(self, stamp):
        with self.lock:
            self.origin_ns = stamp

    def inspect(self, output, pose_samples):
        from gz.msgs10.image_pb2 import Image
        from PIL import Image as PILImage, ImageDraw
        output = Path(output)
        values = from_configuration(self.contract["industrial_targets"])
        result = {"rendered_motion_verified": False, "exact_stamp_rgb_labels": True,
                  "pose_alignment": "latest_preceding_pose_max_0.1_sim_seconds", "agents": {}}
        poses = {row["stamp_ns"]: row for row in pose_samples}
        pose_stamps = np.asarray(sorted(poses), dtype=np.int64)
        comparisons = []
        for spec in self.contract["camera_pairs"]:
            agent = spec["agent_id"]
            findings = []
            for stamp, rgb_raw, label_raw in self.frames[agent]:
                pose_index = int(np.searchsorted(pose_stamps, stamp, side="right")) - 1
                if pose_index < 0 or stamp - int(pose_stamps[pose_index]) > 100_000_000:
                    continue
                pose = poses[int(pose_stamps[pose_index])]
                messages = [Image(), Image()]
                for message, raw in zip(messages, (rgb_raw, label_raw)):
                    message.ParseFromString(raw)
                pair = pair_rgb_instance(rgb_message=messages[0], label_message=messages[1],
                    rgb_topic=spec["rgb"], segmentation_topic=spec["segmentation"],
                    label_topic=spec["segmentation"] + "/labels_map", encoding=GZ8_INSTANCE_ENCODING)
                if not set(map(int, np.unique(pair.object_labels))) <= set(LABELS):
                    raise ValueError("Unexpected moving target instance labels")
                name = f"{agent}_{stamp}"
                (output / (name + "_rgb.pb")).write_bytes(rgb_raw)
                (output / (name + "_labels.private.pb")).write_bytes(label_raw)
                image_path = name + ".png"
                PILImage.fromarray(pair.rgb).save(output / image_path)
                row = {"stamp_ns": stamp, "elapsed_sim_s": (stamp - self.origin_ns) / 1e9,
                       "pose_stamp_ns": pose["stamp_ns"], "positions_enu": pose["positions_enu"],
                       "rgb_png": image_path, "targets": {}}
                for index, identity in enumerate(IDENTITIES):
                    mask = pair.object_labels == 11 + 11 * index
                    ys, xs = np.nonzero(mask)
                    support = int(np.count_nonzero(mask & color_support(pair.rgb, values[index].color)))
                    row["targets"][identity] = {"pixels": len(xs), "color_pixels": support,
                        "centroid_px": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
                        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                        "unclipped": bool(len(xs) and xs.min() > 0 and ys.min() > 0
                                            and xs.max() < mask.shape[1] - 1 and ys.max() < mask.shape[0] - 1)}
                findings.append(row)
            result["agents"][agent] = findings
        for identity in IDENTITIES:
            best = None
            for agent, frames in result["agents"].items():
                for first in frames:
                    a = first["targets"][identity]
                    if a["pixels"] < 16 or a["color_pixels"] < 8 or not a["unclipped"]:
                        continue
                    for second in frames:
                        if second["stamp_ns"] <= first["stamp_ns"]:
                            continue
                        b = second["targets"][identity]
                        if b["pixels"] < 16 or b["color_pixels"] < 8 or not b["unclipped"]:
                            continue
                        displacement = float(np.linalg.norm(np.asarray(first["positions_enu"][identity])
                                                            - second["positions_enu"][identity]))
                        pixel_shift = float(np.linalg.norm(np.asarray(a["centroid_px"]) - b["centroid_px"]))
                        area_ratio = max(a["pixels"], b["pixels"]) / min(a["pixels"], b["pixels"])
                        if (displacement >= .3 and a["mask_sha256"] != b["mask_sha256"]
                                and (pixel_shift >= 5 or area_ratio >= 1.3)):
                            score = displacement
                            if best is None or score > best[0]:
                                best = (score, agent, first, second, pixel_shift, area_ratio)
            if best is None:
                raise ValueError(f"No paired rendered movement proof for {identity}")
            _, agent, first, second, pixel_shift, area_ratio = best
            comparisons.append({"identity": identity, "agent": agent,
                "before": first["rgb_png"], "after": second["rgb_png"],
                "pose_displacement_m": best[0], "centroid_shift_px": pixel_shift, "area_ratio": area_ratio})
        sheet = PILImage.new("RGB", (1024, len(comparisons) * 408), "white")
        for index, comparison in enumerate(comparisons):
            for column, key in enumerate(("before", "after")):
                with PILImage.open(output / comparison[key]) as frame:
                    sheet.paste(frame.resize((512, 384)), (column * 512, index * 408 + 24))
                ImageDraw.Draw(sheet).text((column * 512 + 8, index * 408 + 6),
                    f"{comparison['identity']} / {comparison['agent']} / {key}", fill="black")
        sheet.save(output / "rendered_motion.png")
        result.update({"rendered_motion_verified": True, "comparisons": comparisons,
                       "policy_evaluated": False, "flight_verified": False})
        return result
