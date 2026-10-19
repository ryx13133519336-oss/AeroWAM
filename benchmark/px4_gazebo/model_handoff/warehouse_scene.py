"""Pinned external warehouse adapter. Private, static admission; no flight path."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET
import zipfile

from .annotation_models import AIRCRAFT_MODEL, CAMERA_MODEL, LABEL_SENSOR, verify as verify_cameras
from .search_pilot_scene import _search_objects

COMMIT = "3c23a698bf0b4e366ddf8b084af507c519bd3483"
ARCHIVE_SHA256 = "00cd9528271002b30396e2957da9fb969584c9b18d5b8ad9a8b97fde2c14f214"
RESOURCE_TABLE_SHA256 = "543f66b29de9f055fef3dea7e9a8c6f195667495fb053145b134fe6f172534b6"
SOURCE_URL = "https://github.com/aws-robotics/aws-robomaker-small-warehouse-world"
SCHEMA = "aerowam-external-warehouse/1-draft"
WORLD = "aerowam_warehouse_static"
CASES = ("rack_front", "rack_back")
# Private admission fixtures, not spawn positions or routes for any policy.
CAMERAS = {
    "rack_front": ((1., -5.6, .4, 0., -.15, .7), (1., -4.6, .4, 0., -.15, .6)),
    "rack_back": ((3., -2.14, 1.2, 0., 0., 0.), (3., -.33, 1.2, 0., 0., 0.)),
}
TARGETS = {"probe_red": (5.5, -2.14, 1.2), "probe_blue": (5.5, -.33, 1.2)}
RUNTIME_WORLD = "aerowam_warehouse_startup"
AIRCRAFT_POSES = ((0., -2., .1, 0., 0., 0.), (0., 2., .1, 0., 0., 0.))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_resource(root, relative):
    """Reject traversal, remote URIs, symlink escapes and missing resources."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
        raise ValueError("Resource must be a contained relative path")
    root = Path(root).resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaped resource: {relative}")
    return path


def prepare(archive, destination):
    """Extract only audited resource types from the byte-pinned upstream ZIP.

    Never runs upstream launch files/plugins. Leaves partial output on failure
    for diagnosis; never overwrites an existing package.
    """
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Upstream archive SHA256 mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    prefix = f"aws-robomaker-small-warehouse-world-{COMMIT}/"
    files = {}
    with zipfile.ZipFile(archive) as source:
        world_bytes = source.read(prefix + "worlds/no_roof_small_warehouse.world")
        source_world = ET.fromstring(world_bytes).find("world")
        used_models = {m.findtext("include/uri").removeprefix("model://")
                       for m in source_world.findall("model")}
        for member in source.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            relative = member.filename[len(prefix):]
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
                raise ValueError("Unsafe archive member")
            selected = relative == "LICENSE" or (
                len(path.parts) >= 3 and path.parts[0] == "models" and path.parts[1] in used_models
                and path.suffix.lower() in {".dae", ".png", ".sdf", ".config"}
            ) or relative == "worlds/no_roof_small_warehouse.world"
            if not selected:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            files[relative] = sha256(target)
    manifest = {"schema": SCHEMA, "source_url": SOURCE_URL, "commit": COMMIT,
                "archive_sha256": ARCHIVE_SHA256, "files": files,
                "flight_admitted": False, "license_file": "LICENSE"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify(destination)
    return manifest


def verify(package):
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("commit") != COMMIT
            or manifest.get("archive_sha256") != ARCHIVE_SHA256 or not manifest.get("files")
            or "LICENSE" not in manifest["files"]):
        raise ValueError("Unsupported warehouse package provenance")
    table = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(table).hexdigest() != RESOURCE_TABLE_SHA256:
        raise ValueError("Pinned warehouse resource table mismatch")
    for relative, expected in manifest["files"].items():
        if sha256(local_resource(package, relative)) != expected:
            raise ValueError(f"Warehouse resource changed: {relative}")
    # COLLADA texture paths are relative to their meshes. Preserve the layout;
    # validate every reference without normalizing away an external escape.
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    for relative in manifest["files"]:
        if not relative.lower().endswith(".dae"):
            continue
        mesh = local_resource(package, relative)
        for image in ET.parse(mesh).findall(".//c:library_images/c:image/c:init_from", ns):
            target = (mesh.parent / image.text.strip()).resolve()
            if not target.is_relative_to(package.resolve()) or not target.is_file():
                raise ValueError(f"Missing or escaped COLLADA texture: {image.text}")
            if target.relative_to(package.resolve()).as_posix() not in manifest["files"]:
                raise ValueError("Unfingerprinted COLLADA texture")
    return manifest


def _inline_model(instance, package, manifest):
    uri = instance.findtext("include/uri", "")
    if not uri.startswith("model://") or "/" in uri[len("model://"):]:
        raise ValueError("Only local pinned warehouse model includes are supported")
    relative = f"models/{uri[len('model://'):]}/model.sdf"
    if relative not in manifest["files"]:
        raise ValueError("Unfingerprinted model")
    model = deepcopy(ET.parse(local_resource(package, relative)).getroot().find("model"))
    if model is None or model.findall(".//include") or model.findall(".//plugin"):
        raise ValueError("Nested includes and upstream plugins are not admitted")
    model.set("name", instance.get("name"))
    model.find("static").text = "true"
    # These assets are strictly static. Classic accepted nonphysical inertia
    # tensors that Harmonic rejects even on static links. Omit unused inertial
    # blocks, retaining every collision, surface, mesh scale and visual.
    for link in model.findall(".//link"):
        for inertial in link.findall("inertial"):
            link.remove(inertial)
    for old in model.findall("pose"):
        model.remove(old)
    pose = deepcopy(instance.find("pose"))
    pose.attrib.clear()
    model.insert(0, pose)
    for node in model.findall(".//uri"):
        if not node.text.startswith("model://"):
            raise ValueError("Unexpected upstream resource URI")
        relative = "models/" + node.text[len("model://"):]
        if relative not in manifest["files"]:
            raise ValueError("Unfingerprinted mesh")
        node.text = local_resource(package, relative).as_uri()
    for visual in model.findall(".//visual"):
        for meta in visual.findall("meta"):
            visual.remove(meta)  # Classic GUI visibility layers, not geometry.
    plugin = ET.SubElement(model, "plugin", filename="gz-sim-label-system", name="gz::sim::systems::Label")
    ET.SubElement(plugin, "label").text = "100"  # All background stays private.
    return model


def static_world(package, camera_models, case):
    if case not in CASES:
        raise ValueError("Unknown warehouse viewpoint control")
    package, camera_models = Path(package), Path(camera_models)
    manifest = verify(package)
    cameras = verify_cameras(camera_models)
    original = ET.parse(package / "worlds/no_roof_small_warehouse.world").getroot().find("world")
    # Reuse the verified RGB/label diagnostic environment, removing only its
    # synthetic wall and camera. All imported collisions and texture units stay.
    tree = ET.parse(Path(__file__).with_name("annotation_probe.sdf"))
    scene = tree.getroot().find("world")
    scene.set("name", WORLD)
    for name in ("probe_wall", "probe_camera"):
        scene.remove(scene.find(f"model[@name='{name}']"))
    scene.find("scene/ambient").text = "0.55 0.55 0.55 1"
    imported = []
    for instance in original.findall("model"):
        model = _inline_model(instance, package, manifest)
        scene.append(model)
        imported.append({"name": model.get("name"), "pose": model.findtext("pose"),
                         "collisions": len(model.findall(".//collision")),
                         "visuals": len(model.findall(".//visual"))})
    for light in original.findall("light"):
        scene.append(deepcopy(light))
    for pose in scene.findall(".//pose"):
        pose.attrib.pop("frame", None)
    _search_objects(scene)
    for name, xyz in TARGETS.items():
        target = scene.find(f"model[@name='{name}']")
        target.find("pose").text = " ".join(map(str, (*xyz, 0, 0, 0)))
        # Fit the physical targets inside the approximately 0.9 m rack aisles.
        # Scale body, stripe and collision together; semantic appearances stay.
        for geometry in target.findall(".//geometry"):
            for size in geometry.findall("box/size"):
                size.text = " ".join(str(float(v) * .6) for v in size.text.split())
            for radius in geometry.findall("sphere/radius"):
                radius.text = str(float(radius.text) * .6)
    pairs = []
    for index, pose in enumerate(CAMERAS[case]):
        name = f"annotation_camera_{index}"
        wrapper = ET.SubElement(scene, "model", name=name)
        ET.SubElement(wrapper, "static").text = "true"
        ET.SubElement(wrapper, "pose").text = " ".join(map(str, pose))
        include = ET.SubElement(wrapper, "include", merge="true")
        ET.SubElement(include, "uri").text = (camera_models.resolve() / CAMERA_MODEL / "model.sdf").as_uri()
        prefix = f"/world/{WORLD}/model/{name}/link/camera_link/sensor"
        pairs.append({"agent_id": f"uav_{index}", "rgb": prefix + "/camera/image",
                      "segmentation": prefix + f"/{LABEL_SENSOR}/segmentation"})
    return tree, pairs, {"warehouse": manifest, "cameras": cameras, "imported_models": imported,
                         "scene_case": case, "camera_poses_enu": CAMERAS[case],
                         "private_targets_enu": TARGETS, "role_assignment": None,
                         "conversion": {"inertials": "omitted on static links",
                                        "classic_visual_layers": "omitted", "mesh_units": "unchanged",
                                        "source_world": "no_roof_small_warehouse.world"},
                         "flight_admitted": False}


def aircraft_world(package, camera_models):
    """Dual dynamic aircraft for disarmed integration only; no flight admission.

    Candidate spawns are in the central aisle, not the narrow rack channels.
    No role, waypoint or private target coordinate is passed to a controller.
    """
    tree, _, resources = static_world(package, camera_models, "rack_back")
    scene = tree.getroot().find("world")
    scene.set("name", RUNTIME_WORLD)
    for index in range(2):
        scene.remove(scene.find(f"model[@name='annotation_camera_{index}']"))
    for filename, name in (("user-commands", "UserCommands"), ("contact", "Contact"),
                           ("imu", "Imu"), ("air-pressure", "AirPressure"),
                           ("navsat", "NavSat"), ("magnetometer", "Magnetometer")):
        ET.SubElement(scene, "plugin", filename=f"gz-sim-{filename}-system",
                      name=f"gz::sim::systems::{name}")
    scene.find("physics/max_step_size").text = "0.004"
    ET.SubElement(scene, "gravity").text = "0 0 -9.81"
    # Explicit simulation georeference matching the installed PX4 lawn world;
    # required by Harmonic NavSat/Magnetometer, not a real deployment location.
    spherical = ET.SubElement(scene, "spherical_coordinates")
    for key, value in (("surface_model", "EARTH_WGS84"), ("world_frame_orientation", "ENU"),
                       ("latitude_deg", "47.397971057728974"),
                       ("longitude_deg", "8.546163739800146"), ("elevation", "0")):
        ET.SubElement(spherical, key).text = value
    ET.SubElement(scene, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
    pairs = []
    for index, pose in enumerate(AIRCRAFT_POSES):
        include = ET.SubElement(scene, "include")
        ET.SubElement(include, "uri").text = (Path(camera_models).resolve() / AIRCRAFT_MODEL / "model.sdf").as_uri()
        ET.SubElement(include, "name").text = f"annotation_uav_{index}"
        ET.SubElement(include, "pose").text = " ".join(map(str, pose))
        prefix = f"/world/{RUNTIME_WORLD}/model/annotation_uav_{index}/link/camera_link/sensor"
        pairs.append({"agent_id": f"uav_{index}", "rgb": prefix + "/camera/image",
                      "segmentation": prefix + f"/{LABEL_SENSOR}/segmentation"})
    resources = {**resources, "scene_case": "dual_aircraft_disarmed", "camera_poses_enu": None,
                 "aircraft_spawn_poses_enu": AIRCRAFT_POSES, "world_name": RUNTIME_WORLD,
                 "camera_pairs": pairs, "role_assignment": None, "flight_admitted": False,
                 "readiness": "unarmed_startup_only",
                 "georeference": "PX4_lawn_simulation_reference",
                 "remaining_flight_gates": ["aircraft_clearance", "physical_contact_detection",
                                            "scene_bound_controller", "role_free_motion_acceptance"]}
    return tree, resources
