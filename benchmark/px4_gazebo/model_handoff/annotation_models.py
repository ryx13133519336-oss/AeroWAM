"""Generate opt-in camera resources; never edit installed PX4 models.

The generated aircraft must be spawned explicitly and attached with
PX4_GZ_MODEL_NAME in a future runner. PX4's gz_env.sh resets PX4_GZ_MODELS,
so simply exporting a different model directory is not a supported launch.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CAMERA_MODEL = "aerowam_mono_cam_annotations_v1"
AIRCRAFT_MODEL = "aerowam_x500_annotations_v1"
LABEL_SENSOR = "private_instance"


def paired_camera(source: ET.Element) -> ET.Element:
    """Copy RGB sensor and add exact intrinsics/pose/rate-matched labels."""
    result = deepcopy(source)
    model = result.find("model")
    if model is None or model.get("name") != "mono_cam":
        raise ValueError("Expected installed mono_cam model")
    sensors = model.findall(".//sensor")
    if len(sensors) != 1 or sensors[0].get("type") != "camera" or sensors[0].get("name") != "camera":
        raise ValueError("Expected exactly one ordinary camera sensor")
    rgb = sensors[0]
    if rgb.find("topic") is not None:
        raise ValueError("Explicit RGB topic would collide between two cameras")
    camera = rgb.find("camera")
    if camera is None or camera.find("image") is None or camera.find("horizontal_fov") is None:
        raise ValueError("Camera intrinsics missing")
    if any(camera.find(name) is not None for name in ("noise", "distortion")):
        raise ValueError("Noise/distortion needs a separately verified segmentation contract")
    width, height = (int(camera.findtext("image/" + name, "0")) for name in ("width", "height"))
    if not 0 < width <= 4096 or not 0 < height <= 4096 or float(rgb.findtext("update_rate", "0")) <= 0:
        raise ValueError("Invalid image size/update rate")
    model.set("name", CAMERA_MODEL)
    labels = deepcopy(rgb)
    labels.set("name", LABEL_SENSOR)
    labels.set("type", "segmentation")
    ET.SubElement(labels.find("camera"), "segmentation_type").text = "instance"
    visualization = labels.find("visualize")
    if visualization is None:
        visualization = ET.SubElement(labels, "visualize")
    visualization.text = "false"
    link = next(link for link in model.findall("link") if rgb in list(link))
    link.append(labels)
    return result


def generate(px4_root: Path, output: Path) -> dict:
    source_dir = px4_root.resolve() / "Tools/simulation/gz/models"
    paths = {"camera": source_dir / "mono_cam/model.sdf",
             "aircraft": source_dir / "x500_mono_cam/model.sdf",
             "base": source_dir / "x500/model.sdf"}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    camera = paired_camera(ET.fromstring(raw["camera"]))
    aircraft = ET.fromstring(raw["aircraft"])
    model = aircraft.find("model")
    if model is None or model.get("name") != "x500_mono_cam":
        raise ValueError("Expected installed x500_mono_cam")
    model.set("name", AIRCRAFT_MODEL)
    uris = model.findall("include/uri")
    if {uri.text for uri in uris} != {"x500", "model://mono_cam"} or len(uris) != 2:
        raise ValueError("Unknown aircraft include structure")
    # Local absolute references avoid resource-path precedence and never
    # shadow the installed mono_cam name. Output is local, not relocatable.
    for uri in uris:
        uri.text = paths["base"].as_uri() if uri.text == "x500" else (output.resolve() / CAMERA_MODEL / "model.sdf").as_uri()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "aerowam-annotation-models/1-draft", "flight_ready": False,
                "camera_model": CAMERA_MODEL, "aircraft_model": AIRCRAFT_MODEL,
                "label_sensor": LABEL_SENSOR,
                "source_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()},
                "files": {}}
    for name, root in ((CAMERA_MODEL, camera), (AIRCRAFT_MODEL, aircraft)):
        folder = output / name
        folder.mkdir()
        ET.ElementTree(root).write(folder / "model.sdf", encoding="utf-8", xml_declaration=True)
        config = ET.Element("model")
        ET.SubElement(config, "name").text = name
        ET.SubElement(config, "version").text = "1.0"
        ET.SubElement(config, "sdf", version="1.9").text = "model.sdf"
        ET.SubElement(config, "description").text = "Opt-in AeroWAM private annotation camera; draft"
        ET.ElementTree(config).write(folder / "model.config", encoding="utf-8", xml_declaration=True)
        for filename in ("model.sdf", "model.config"):
            relative = f"{name}/{filename}"
            manifest["files"][relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {f"{name}/{file}" for name in (CAMERA_MODEL, AIRCRAFT_MODEL) for file in ("model.sdf", "model.config")}
    if manifest.get("schema") != "aerowam-annotation-models/1-draft" or set(manifest.get("files", {})) != expected:
        raise ValueError("Unexpected annotation resource manifest")
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Annotation resource fingerprint mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.px4_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
