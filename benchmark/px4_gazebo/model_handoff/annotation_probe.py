"""Isolated static Gazebo transport diagnostic, not a flight/evaluation run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from .annotation_models import CAMERA_MODEL, LABEL_SENSOR, verify


def run(output: Path, timeout_s: float = 40., scene_case: str = "partial", rig_models: Path | None = None) -> dict:
    if not 1 <= timeout_s <= 60:
        raise ValueError("timeout must be 1..60 seconds")
    from .search_pilot_scene import OCCLUSION_CASES, static_occlusion_world
    if scene_case not in {"partial", "clear", "full", *OCCLUSION_CASES}:
        raise ValueError("Unknown static scene case")
    if scene_case in OCCLUSION_CASES and rig_models is None:
        raise ValueError("Search occlusion diagnostic requires paired camera models")
    output.mkdir(parents=True, exist_ok=False)
    partition = "aerowam_annotation_" + uuid.uuid4().hex
    previous = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = partition
    process = None
    node = None
    summary = {"schema": "aerowam-annotation-probe/1-draft", "partition": partition,
               "scene_case": scene_case,
               "topics": [], "messages": {}, "capture_pair_verified": False,
               "live_multitask_ready": False}
    lock = threading.Lock()
    received = {}
    subscriptions = set()
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        node = Node()
        tree = ET.parse(Path(__file__).with_name("annotation_probe.sdf"))
        wall = tree.getroot().find("world/model[@name='probe_wall']")
        if scene_case == "clear":
            wall.find("pose").text = "3 20 1 0 0 0"
        elif scene_case == "full":
            wall.find("pose").text = "3 0.6 1 0 0 0"
            wall.find("link/visual/geometry/box/size").text = "0.1 1.4 1.5"
        pairs = [{"agent_id": "uav_0", "rgb": "/aerowam/annotation_probe/rgb",
                  "segmentation": "/aerowam/annotation_probe/segmentation"}]
        if scene_case in OCCLUSION_CASES:
            summary["model_resources"] = verify(rig_models)
            tree, pairs = static_occlusion_world(rig_models, scene_case)
        elif rig_models is not None:
            summary["model_resources"] = verify(rig_models)
            scene = tree.getroot().find("world")
            scene.remove(scene.find("model[@name='probe_camera']"))
            pairs = []
            for index, y in enumerate((0., -.6)):
                # Fixed cameras only: this is a dual-view native-resolution
                # sensor check, not a flight or moving-pose calibration test.
                name = f"annotation_camera_{index}"
                wrapper = ET.SubElement(scene, "model", name=name)
                ET.SubElement(wrapper, "static").text = "true"
                ET.SubElement(wrapper, "pose").text = f"0 {y} 1 0 0 0"
                include = ET.SubElement(wrapper, "include", merge="true")
                ET.SubElement(include, "uri").text = (rig_models.resolve() / CAMERA_MODEL / "model.sdf").as_uri()
                prefix = f"/world/aerowam_annotation_probe/model/{name}/link/camera_link/sensor"
                pairs.append({"agent_id": f"uav_{index}", "rgb": prefix + "/camera/image",
                              "segmentation": prefix + f"/{LABEL_SENSOR}/segmentation"})
        summary["camera_pairs"] = pairs
        world = output / "world.sdf"
        tree.write(world, encoding="utf-8", xml_declaration=True)
        summary["world_sha256"] = hashlib.sha256(world.read_bytes()).hexdigest()
        with (output / "gazebo.log").open("xb") as log:
            process = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(world)],
                                       env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Gazebo exited {process.returncode}")
                listed = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=5., check=False)
                candidates = set(listed.stdout.splitlines())
                expected = {topic for pair in pairs for topic in
                            (pair["rgb"], pair["segmentation"] + "/labels_map", pair["segmentation"] + "/colored_map")}
                topics = sorted(candidates & expected)
                summary["topics"] = topics
                summary["discovered_image_topics"] = sorted(t for t in candidates if "image" in t or "map" in t)
                for topic in topics:
                    if topic in subscriptions or "camera_info" in topic:
                        continue
                    def callback(message, key=topic):
                        with lock:
                            values = received.setdefault(key, {})
                            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec
                            if stamp not in values and len(values) < 8:
                                snapshot = Image()
                                snapshot.CopyFrom(message)
                                values[stamp] = snapshot
                    if node.subscribe(Image, topic, callback):
                        subscriptions.add(topic)
                with lock:
                    # Stop once streams have enough frames, but no acceptance
                    # claim until payload encoding and pairing are checked.
                    enough = all(len(set(received.get(pair["rgb"], {})) &
                                     set(received.get(pair["segmentation"] + "/labels_map", {}))) >= 3 for pair in pairs)
                if enough:
                    break
                time.sleep(.2)
            with lock:
                for index, (topic, rows) in enumerate(sorted(received.items())):
                    metadata = []
                    for stamp, message in sorted(rows.items()):
                        filename = f"stream_{index}_{stamp}.pb"
                        (output / filename).write_bytes(message.SerializeToString())
                        metadata.append({"stamp_ns": stamp, "width": message.width, "height": message.height,
                                         "step": message.step, "format": message.pixel_format_type,
                                         "bytes": len(message.data), "file": filename})
                    summary["messages"][topic] = metadata
                summary["pair_counts"] = {pair["agent_id"]: len(set(received.get(pair["rgb"], {})) &
                                                set(received.get(pair["segmentation"] + "/labels_map", {}))) for pair in pairs}
            if min(summary["pair_counts"].values()) < 3:
                raise TimeoutError("Fewer than three exact-stamp RGB/labels_map pairs; inspect gazebo.log")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if node is not None:
            for topic in subscriptions:
                try:
                    node.unsubscribe(topic)
                except Exception as exc:
                    summary.setdefault("cleanup_errors", []).append(f"Unsubscribe {topic}: {exc}")
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.)
                except ProcessLookupError:
                    process.wait(timeout=5.)
            except ProcessLookupError:
                process.wait(timeout=5.)
        if previous is None:
            os.environ.pop("GZ_PARTITION", None)
        else:
            os.environ["GZ_PARTITION"] = previous
        summary["owned_process_stopped"] = process is None or process.poll() is not None
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=40.)
    parser.add_argument("--scene-case", choices=("partial", "clear", "full"), default="partial")
    parser.add_argument("--rig-models", type=Path, help="Generated opt-in models; test two native-resolution camera rigs")
    args = parser.parse_args()
    result = run(args.output_dir, args.timeout_s, args.scene_case, args.rig_models)
    compact = {key: value for key, value in result.items() if key != "messages"}
    compact["stream_frames"] = {topic: len(rows) for topic, rows in result["messages"].items()}
    print(json.dumps(compact, indent=2))
    raise SystemExit(2 if "error" in result or result.get("cleanup_errors") or not result["owned_process_stopped"] else 0)


if __name__ == "__main__":
    main()
