"""Offline SubT Urban local scene; private geometry, no prescribed UAV roles."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import zipfile

from .annotation_models import AIRCRAFT_MODEL, CAMERA_MODEL, LABEL_SENSOR, verify as verify_cameras
from .warehouse_scene import local_resource, sha256
from .industrial_targets import POSITIONS, components, install, instrument_contacts, fingerprint

SCHEMA = "aerowam-subt-urban/1-draft"
WORLD = "aerowam_subt_urban"
RESOURCE_TABLE_SHA256 = "fa789cf8ebbb0f2c47caf094993d5675bbf6c201c95844d4a6798fe0e305b7f7"
ASSETS = {
    "urban_2_story": {"name": "Urban 2 Story", "version": 14,
                      "sha256": "db8a1368e60acb9d3c15a68d744baf523c020a752ae2357a8309fde7cc2a6235"},
    "urban_service_straight": {"name": "Urban Service Room Straight", "version": 6,
                               "sha256": "5edc5e0d2e61d2f985b03676eb12d4a40e0fd6beb650ea55dae3d3842300a08a"},
    "urban_stairwell": {"name": "Urban Stairwell Platform Centered", "version": 14,
                        "sha256": "beda9e47a006b3c93dc98570a3054b272cfb5b94435ba2cefac899d314f436e6"},
    "urban_platform": {"name": "Urban Platform", "version": 3,
                       "sha256": "fc6149685a88e054daaf2e03d7489250d115bead6d123665e9f3349f8cf69406"},
}
# V1 imports the complete two-story tile, already containing stairs, shaft,
# connecting corridors and lower/upper rooms. Extra downloaded tiles are not
# silently placed without validating their physical connections.
CAMERAS = {
    "overview": ((32., -35., 27., 0., .48, 2.3), (0., -8., 2., 0., 0., 0.)),
    "hidden": ((0., -12., 2., 0., 0., .56), (0., -14., 2., 0., 0., .72)),
    "revealed": ((12., -8., 2., 0., 0., 2.90), (12., -6., 2., 0., 0., -2.90)),
    "upper": ((0., 0., 12., 0., 0., 0.), (10., -10., 12., 0., 0., 2.3)),
}
TARGETS = POSITIONS
SPAWNS = ((0., -2., .15, 0., 0., 0.), (0., 2., .15, 0., 0., 0.))


def prepare(source, package):
    """Prepare pinned assets without invoking any upstream scripts or plugins."""
    source, package = Path(source), Path(package)
    # Validate before creating output, not after extracting an unknown archive.
    for key, asset in ASSETS.items():
        if sha256(source / (key + ".zip")) != asset["sha256"]:
            raise ValueError(f"Pinned archive mismatch: {key}")
    package.mkdir(parents=True, exist_ok=False)
    for key, asset in ASSETS.items():
        folder = package / key
        folder.mkdir()
        with zipfile.ZipFile(source / (key + ".zip")) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or ":" in str(name)
                        or "\\" in str(name) or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unsafe archive resource")
                if member.file_size > 200_000_000:
                    raise ValueError("Oversized archive member")
                target = folder / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as incoming, target.open("xb") as outgoing:
                    import shutil
                    shutil.copyfileobj(incoming, outgoing)
        # Preserve upstream attribution, model.config and metadata.pbtxt, plus
        # the Fuel license declaration. Source ZIPs remain outside the package.
        metadata = json.loads((source / (key + ".metadata.json")).read_text(encoding="utf-8-sig"))
        if (metadata.get("name") != asset["name"] or metadata.get("version") != asset["version"]
                or metadata.get("owner") != "OpenRobotics"
                or metadata.get("license_name") != "Creative Commons Attribution 4.0 International"):
            raise ValueError("Unexpected asset identity or license")
        (folder / "fuel_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    # COLLADA embeds moving `tip` texture URLs. Rewrite copies only, pinning
    # every reference to the downloaded shared texture package.
    for mesh in package.glob("*/meshes/*.dae"):
        root = ET.parse(mesh)
        # gz-common's COLLADA reader expects an unprefixed COLLADA root.
        ET.register_namespace("", "http://www.collada.org/2005/11/COLLADASchema")
        changed = False
        for item in root.findall(".//{*}library_images/{*}image/{*}init_from"):
            target = texture_resource(package, item.text)
            import os
            item.text = Path(os.path.relpath(target, mesh.parent)).as_posix()
            changed = True
        if changed:
            root.write(mesh, encoding="utf-8", xml_declaration=True)
    (package / "ATTRIBUTION.txt").write_text(
        "SubT Urban assets by Open Robotics; authors preserved in each model.config and metadata.pbtxt.\n"
        "Licensed under Creative Commons Attribution 4.0 International:\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Source: https://app.gazebosim.org/OpenRobotics/fuel\n"
        "AeroWAM modifications: offline texture paths; stripped legacy scripts/detectors in generated SDF; "
        "local lighting, diagnostic targets and sensor rigs. Original geometry retained.\n", encoding="utf-8", newline="\n")
    files = {p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()}
    manifest = {"schema": SCHEMA, "assets": ASSETS, "files": files,
                "layout": "single_complete_two_story_tile", "flight_admitted": False}
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify(package)
    return manifest


def texture_resource(package, uri):
    parsed = urlparse(uri.strip())
    if parsed.scheme:
        prefix = "/1.0/OpenRobotics/models/Urban Platform/"
        decoded = unquote(parsed.path)
        if (parsed.scheme != "https" or parsed.netloc not in {"fuel.ignitionrobotics.org", "fuel.gazebosim.org"}
                or not any(decoded.startswith(prefix + version + "/files/materials/textures/")
                           for version in ("tip", "3")) or parsed.query or parsed.fragment):
            raise ValueError("Unapproved remote texture dependency")
        relative = decoded.split("/files/", 1)[1]
    else:
        raise ValueError("Expected a pinned Urban Platform texture URL")
    return local_resource(Path(package) / "urban_platform", relative)


def verify(package):
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("assets") != ASSETS or not manifest.get("files"):
        raise ValueError("Unexpected SubT provenance")
    # Fuel counters / update times change on download. Geometry, original
    # attribution and converted resources are pinned independently of those.
    table = {k: v for k, v in manifest["files"].items() if not k.endswith("/fuel_metadata.json")}
    table_digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if table_digest != RESOURCE_TABLE_SHA256:
        raise ValueError("Pinned SubT resource table mismatch")
    for relative, digest in manifest["files"].items():
        if sha256(local_resource(package, relative)) != digest:
            raise ValueError(f"SubT resource fingerprint mismatch: {relative}")
    for mesh in package.glob("*/meshes/*.dae"):
        for item in ET.parse(mesh).findall(".//{*}library_images/{*}image/{*}init_from"):
            target = (mesh.parent / item.text.strip()).resolve()
            if (not target.is_relative_to(package.resolve()) or not target.is_file()
                    or target.relative_to(package.resolve()).as_posix() not in manifest["files"]):
                raise ValueError("Missing or unfingerprinted COLLADA texture")
    return manifest


def model_copy(package, key, name):
    folder = Path(package) / key
    model = deepcopy(ET.parse(folder / "model.sdf").getroot().find("model"))
    if model is None or model.findall(".//include"):
        raise ValueError("Unexpected nested model dependency")
    model.set("name", name)
    for parent in model.iter():
        for plugin in list(parent.findall("plugin")):
            if (plugin.get("filename") != "libignition-gazebo-performer-detector-system.so"
                    or plugin.get("name") != "ignition::gazebo::systems::PerformerDetector"):
                raise ValueError("Unapproved upstream plugin")
            parent.remove(plugin)
        for inertial in list(parent.findall("inertial")):
            parent.remove(inertial)
    for material in model.findall(".//material"):
        for script in material.findall("script"):
            material.remove(script)  # PBR is authored upstream; do not use Classic fallback.
        for item in material.iter():
            if item.tag.endswith("_map") and item.text:
                item.text = texture_resource(package, item.text).as_uri()
    for uri in model.findall(".//mesh/uri"):
        uri.text = local_resource(folder, uri.text).as_uri()
    if model.find("pose") is not None:
        raise ValueError("Unexpected upstream model transform")
    if model.findtext("static") not in {"true", "1"}:
        raise ValueError("Only static SubT assets admitted")
    ET.SubElement(model, "pose").text = "0 0 0 0 0 0"
    plugin = ET.SubElement(model, "plugin", filename="gz-sim-label-system", name="gz::sim::systems::Label")
    ET.SubElement(plugin, "label").text = "100"
    return model


def scene(package, models, *, case="overview", aircraft=False, target_components=None):
    """Returns tree and private contract. Does not start processes or policy."""
    if case not in CAMERAS:
        raise ValueError("Unknown SubT diagnostic view")
    target_components = components(target_components)
    manifest, cameras = verify(package), verify_cameras(Path(models))
    tree = ET.parse(Path(__file__).with_name("annotation_probe.sdf"))
    world = tree.getroot().find("world")
    world.set("name", WORLD)
    for key in ("probe_wall", "probe_camera"):
        world.remove(world.find(f"model[@name='{key}']"))
    target_config = install(world, target_components)
    world.append(model_copy(package, "urban_2_story", "subt_two_story"))
    contacts = instrument_contacts(world)
    ET.SubElement(world, "plugin", filename="gz-sim-user-commands-system", name="gz::sim::systems::UserCommands")
    ET.SubElement(world, "plugin", filename="gz-sim-contact-system", name="gz::sim::systems::Contact")
    world.find("scene/ambient").text = "0.35 0.35 0.35 1"
    # Controlled neutral fill lights, not claimed to reproduce SubT darkness.
    for index, xyz in enumerate(((0, 0, 8), (10, -10, 8), (-10, 10, 8), (0, 0, 18), (10, -10, 18), (-10, 10, 18))):
        light = ET.SubElement(world, "light", name=f"local_fill_{index}", type="point")
        ET.SubElement(light, "pose").text = " ".join(map(str, (*xyz, 0, 0, 0)))
        ET.SubElement(light, "diffuse").text = ".8 .8 .8 1"
        attenuation = ET.SubElement(light, "attenuation")
        for tag, value in (("range", "35"), ("constant", "1"), ("linear", ".02"), ("quadratic", ".001")):
            ET.SubElement(attenuation, tag).text = value
    gui = ET.SubElement(world, "gui", fullscreen="false")
    view = ET.SubElement(gui, "plugin", filename="MinimalScene", name="3D View")
    ET.SubElement(view, "engine").text = "ogre2"
    ET.SubElement(view, "scene").text = "scene"
    ET.SubElement(view, "camera_pose").text = "35 -42 32 0 .5 2.25"
    ET.SubElement(gui, "plugin", filename="GzSceneManager", name="Scene Manager")
    ET.SubElement(gui, "plugin", filename="InteractiveViewControl", name="View Control")
    ET.SubElement(gui, "plugin", filename="CameraTracking", name="Camera Tracking")
    pairs = []
    if aircraft:
        for filename, name in (("imu", "Imu"),
                               ("air-pressure", "AirPressure"), ("navsat", "NavSat"), ("magnetometer", "Magnetometer")):
            ET.SubElement(world, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
        world.find("physics/max_step_size").text = ".004"
        ET.SubElement(world, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
        coords = ET.SubElement(world, "spherical_coordinates")
        for key, value in (("surface_model", "EARTH_WGS84"), ("world_frame_orientation", "ENU"),
                           ("latitude_deg", "47.397971057728974"), ("longitude_deg", "8.546163739800146"), ("elevation", "0")):
            ET.SubElement(coords, key).text = value
    for index, pose in enumerate(SPAWNS if aircraft else CAMERAS[case]):
        name = f"annotation_uav_{index}" if aircraft else f"annotation_camera_{index}"
        if aircraft:
            include = ET.SubElement(world, "include")
            ET.SubElement(include, "name").text = name
        else:
            wrapper = ET.SubElement(world, "model", name=name)
            ET.SubElement(wrapper, "static").text = "true"
            ET.SubElement(wrapper, "pose").text = " ".join(map(str, pose))
            include = ET.SubElement(wrapper, "include", merge="true")
        ET.SubElement(include, "uri").text = (Path(models).resolve() / (AIRCRAFT_MODEL if aircraft else CAMERA_MODEL) / "model.sdf").as_uri()
        if aircraft:
            ET.SubElement(include, "pose").text = " ".join(map(str, pose))
        prefix = f"/world/{WORLD}/model/{name}/link/camera_link/sensor"
        pairs.append({"agent_id": f"uav_{index}", "rgb": prefix + "/camera/image",
                      "segmentation": prefix + f"/{LABEL_SENSOR}/segmentation"})
    return tree, {"schema": SCHEMA, "world_name": WORLD, "subt": manifest, "cameras": cameras,
                  "camera_pairs": pairs, "camera_poses_enu": None if aircraft else CAMERAS[case],
                  "aircraft_spawn_poses_enu": SPAWNS if aircraft else None,
                  "private_targets_enu": TARGETS, "role_assignment": None, "flight_admitted": False,
                  "industrial_targets": target_config, "target_config_sha256": fingerprint(target_components),
                  "contact_sensors": contacts,
                  "policy_evaluated": False, "layout": "single_complete_two_story_tile",
                  "case": case, "collision_meshes": 1, "lighting": "declared_neutral_fill",
                  "readiness": "disarmed_integration_candidate" if aircraft else "static_render_candidate",
                  "motion_commands_sent": 0}
