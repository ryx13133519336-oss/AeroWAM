"""One private paired search scene. A pilot, not a compositional split."""

from copy import deepcopy
import hashlib
import math
import random
from pathlib import Path
import xml.etree.ElementTree as ET

from .annotation_models import CAMERA_MODEL, LABEL_SENSOR, verify
from .annotation_session_smoke import build_world
from .fixtures import Appearance
from .language_tasks import LanguageTask, paired_target_tasks, role_free_task

WORLD = "aerowam_search_pilot"
OBJECTS = {"probe_red": Appearance("red", "sphere", "black"),
           "probe_blue": Appearance("blue", "cube", "white")}
LABELS = {0: None, 11: "probe_red", 22: "probe_blue"}
SPAWNS_NED = ((0., 0., 0.), (-3., 0., 0.))  # SDF ENU (0,0), (0,-3).
OCCLUSION_CASES = ("search_clear", "search_blocked", "search_revealed")
OCCLUSION_WORLD = "aerowam_search_occlusion_probe"


def manual_condition(instruction, expected_color, *, role_mode="fixed"):
    """Human-authored text + separate private scoring annotation.

    Do not infer the evaluator answer using the policy's color parser.
    The human must check that this text describes the selected target.
    """
    if expected_color not in {"red", "blue"}:
        raise ValueError("Choose the private expected red/blue object")
    descriptor = next(value for value in OBJECTS.values() if value.color == expected_color)
    if role_mode == "unassigned":
        return role_free_task("search", descriptor, instruction=instruction).bind(OBJECTS)
    if role_mode != "fixed":
        raise ValueError("Unsupported role mode")
    return LanguageTask("search", instruction, descriptor).bind(OBJECTS)


def paired_conditions(seed=0, *, role_mode="fixed"):
    """Randomized instruction order, private answers, one scene-level group.

    Identical scene and initial conditions must be recreated per condition.
    Each method gets this same order; opaque run IDs must not encode answers.
    """
    conditions = list(paired_target_tasks("search", OBJECTS, role_mode=role_mode))
    random.Random(seed).shuffle(conditions)
    return tuple(conditions)


def search_world(px4_root, models):
    tree = build_world(px4_root, models)
    scene = tree.getroot().find("world")
    scene.set("name", WORLD)
    scene.remove(scene.find("model[@name='probe_wall']"))
    for index, y in enumerate((0., -3.)):
        # Require a view change: targets begin outside UAV0's forward view.
        scene.find(f"include[name='annotation_uav_{index}']/pose").text = f"0 {y} 0.1 0 0 1.6"
    _search_objects(scene)
    return tree


def _search_objects(scene):
    """Use identical object assets in the flight pilot and static diagnostic."""
    for identity, y in (("probe_red", 1.), ("probe_blue", -1.)):
        scene.find(f"model[@name='{identity}']/pose").text = f"4 {y} 2.4 0 0 0"
    # The older diagnostic cube has no stripe; do not author a false instruction.
    blue = scene.find("model[@name='probe_blue']/link")
    stripe = deepcopy(scene.find("model[@name='probe_red']/link/visual[@name='stripe']"))
    stripe.find("geometry/box/size").text = "0.82 0.16 0.82"
    stripe.find("material/ambient").text = "0.95 0.95 0.95 1"
    stripe.find("material/diffuse").text = "0.95 0.95 0.95 1"
    blue.append(stripe)
    # Two target collisions for independent safety checks in this pilot.
    for identity in OBJECTS:
        link = scene.find(f"model[@name='{identity}']/link")
        collision = ET.SubElement(link, "collision", name="body_collision")
        collision.append(deepcopy(link.find("visual[@name='body']/geometry")))
        sensor = ET.SubElement(link, "sensor", name="private_contact", type="contact")
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "update_rate").text = "30"
        ET.SubElement(sensor, "topic").text = f"/aerowam/search_pilot/private/contacts/{identity}"
        ET.SubElement(ET.SubElement(sensor, "contact"), "collision").text = "body_collision"


def static_occlusion_world(models, scene_case):
    """Matched static viewpoints only; never spawns aircraft or PX4.

    Clear/blocked have identical camera and target poses. Revealed retains
    the same wall and targets, changing only the two camera poses. These are
    independent captures, not a temporal reacquisition or memory experiment.
    Nothing here changes the default flight scene or supplies policy inputs.
    """
    if scene_case not in OCCLUSION_CASES:
        raise ValueError("Unknown search occlusion case")
    models = Path(models)
    verify(models)
    tree = ET.parse(Path(__file__).with_name("annotation_probe.sdf"))
    scene = tree.getroot().find("world")
    scene.set("name", OCCLUSION_WORLD)
    scene.remove(scene.find("model[@name='probe_camera']"))
    wall = scene.find("model[@name='probe_wall']")
    if scene_case == "search_clear":
        scene.remove(wall)
    else:
        wall.find("pose").text = "2 -0.5 2.4 0 0 0"
        wall.find("link/visual/geometry/box/size").text = "0.15 4 2.8"
        collision = ET.SubElement(wall.find("link"), "collision", name="wall_collision")
        collision.append(deepcopy(wall.find("link/visual/geometry")))
    _search_objects(scene)
    pairs = []
    positions = (6., -8.) if scene_case == "search_revealed" else (0., -3.)
    for index, y in enumerate(positions):
        name = f"annotation_camera_{index}"
        wrapper = ET.SubElement(scene, "model", name=name)
        ET.SubElement(wrapper, "static").text = "true"
        yaw = math.atan2(-y, 4.)
        ET.SubElement(wrapper, "pose").text = f"0 {y} 2.4 0 0 {yaw:.12g}"
        include = ET.SubElement(wrapper, "include", merge="true")
        ET.SubElement(include, "uri").text = (models.resolve() / CAMERA_MODEL / "model.sdf").as_uri()
        prefix = f"/world/{OCCLUSION_WORLD}/model/{name}/link/camera_link/sensor"
        pairs.append({"agent_id": f"uav_{index}", "rgb": prefix + "/camera/image",
                      "segmentation": prefix + f"/{LABEL_SENSOR}/segmentation"})
    return tree, pairs


def scene_digest(tree):
    return hashlib.sha256(ET.tostring(tree.getroot(), encoding="utf-8")).hexdigest()
