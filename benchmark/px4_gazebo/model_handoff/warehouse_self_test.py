"""Warehouse import boundary and collision preservation regressions; no Gazebo."""

from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from .warehouse_scene import (
    ARCHIVE_SHA256, CASES, COMMIT, SCHEMA, _inline_model, local_resource, prepare, sha256,
    AIRCRAFT_POSES, RUNTIME_WORLD, aircraft_world, static_world, verify,
)
from .warehouse_startup import export


class WarehouseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "models/rack/model.sdf"
        self.model.parent.mkdir(parents=True)
        self.mesh = self.root / "models/rack/mesh.DAE"
        self.mesh.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"/>', encoding="utf-8")
        self.model.write_text('''<sdf version="1.6"><model name="rack"><static>1</static>
          <link name="link"><inertial><mass>1000</mass></inertial>
          <collision name="rack_collision"><geometry><mesh><uri>model://rack/mesh.DAE</uri>
          <scale>1 1 1</scale></mesh></geometry><surface><friction><ode><mu>0.2</mu></ode></friction></surface></collision>
          <visual name="rack_visual"><geometry><mesh><uri>model://rack/mesh.DAE</uri></mesh></geometry>
          <meta><layer>1</layer></meta></visual></link></model></sdf>''', encoding="utf-8")
        (self.root / "LICENSE").write_text("test fixture license", encoding="utf-8")
        self.manifest = {"schema": SCHEMA, "commit": COMMIT, "archive_sha256": ARCHIVE_SHA256,
                         "files": {p.relative_to(self.root).as_posix(): sha256(p)
                                   for p in (self.mesh, self.model, self.root / "LICENSE")}}
        self.save_manifest()
        self.instance = ET.fromstring('''<model name="rack_001"><include><uri>model://rack</uri></include>
                                       <pose frame="">4 2 0 0 0 1</pose></model>''')

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        table = json.dumps(self.manifest["files"], sort_keys=True, separators=(",", ":")).encode()
        if hasattr(self, "table_patch"):
            self.table_patch.stop()
        self.table_patch = patch("benchmark.px4_gazebo.model_handoff.warehouse_scene.RESOURCE_TABLE_SHA256",
                                 hashlib.sha256(table).hexdigest())
        self.table_patch.start()
        self.addCleanup(self.table_patch.stop)

    def test_conversion_preserves_collision_surface_scale_and_pose(self):
        source = ET.parse(self.model).getroot().find("model")
        before = deepcopy(source.find("link/collision"))
        result = _inline_model(self.instance, self.root, self.manifest)
        collision = result.find("link/collision")
        before.find("geometry/mesh/uri").text = self.mesh.as_uri()
        self.assertEqual(ET.tostring(collision), ET.tostring(before))
        self.assertEqual(result.findtext("pose"), "4 2 0 0 0 1")
        self.assertEqual(result.findtext("static"), "true")
        self.assertFalse(result.findall(".//inertial"))
        self.assertFalse(result.findall(".//meta"))
        self.assertEqual(result.findtext("plugin/label"), "100")
        self.assertEqual(source.findtext("link/collision/geometry/mesh/uri"), "model://rack/mesh.DAE")

    def test_upstream_plugins_and_nested_includes_rejected(self):
        for extra in ('<plugin name="unsafe" filename="unsafe"/>', '<include><uri>https://example.org</uri></include>'):
            original = self.model.read_text(encoding="utf-8")
            self.model.write_text(original.replace("</model>", extra + "</model>"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plugins"):
                _inline_model(self.instance, self.root, self.manifest)
            self.model.write_text(original, encoding="utf-8")

    def test_manifest_tamper_and_missing_resource_rejected(self):
        verify(self.root)
        self.mesh.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify(self.root)
        self.mesh.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify(self.root)

    def test_resource_traversal_remote_and_windows_paths_rejected(self):
        for relative in ("../LICENSE", "/etc/passwd", "https://example.org/model", "C:/model", "models\\rack"):
            with self.assertRaises(ValueError):
                local_resource(self.root, relative)

    def test_unfingerprinted_and_external_textures_rejected(self):
        for uri in ("https://example.org/texture.png", "../../../outside.png", "missing.png"):
            self.mesh.write_text(f'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
                <library_images><image><init_from>{uri}</init_from></image></library_images></COLLADA>''', encoding="utf-8")
            self.manifest["files"]["models/rack/mesh.DAE"] = sha256(self.mesh)
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, "texture"):
                verify(self.root)

    def test_wrong_archive_rejected_before_output_creation(self):
        archive = self.root / "wrong.zip"
        archive.write_bytes(b"not the pinned upstream archive")
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare(archive, output)
        self.assertFalse(output.exists())

    def test_rehashed_tampered_manifest_is_not_trusted(self):
        self.mesh.write_text("changed and rehashed", encoding="utf-8")
        self.manifest["files"]["models/rack/mesh.DAE"] = sha256(self.mesh)
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resource table"):
            verify(self.root)

    def build_controls(self):
        folder = self.root / "worlds"
        folder.mkdir()
        root = ET.Element("sdf", version="1.6")
        world = ET.SubElement(root, "world", name="default")
        world.append(deepcopy(self.instance))
        ET.ElementTree(root).write(folder / "no_roof_small_warehouse.world")
        with patch("benchmark.px4_gazebo.model_handoff.warehouse_scene.verify", return_value=self.manifest), \
                patch("benchmark.px4_gazebo.model_handoff.warehouse_scene.verify_cameras", return_value={}):
            return [static_world(self.root, self.root / "cameras", case) for case in CASES]

    def test_controls_only_change_camera_poses_no_assigned_roles(self):
        controls = self.build_controls()
        normalized = []
        for tree, pairs, resources in controls:
            self.assertIsNone(resources["role_assignment"])
            self.assertFalse(resources["flight_admitted"])
            self.assertEqual({p["agent_id"] for p in pairs}, {"uav_0", "uav_1"})
            self.assertEqual(resources["imported_models"][0]["collisions"], 1)
            scene = tree.getroot().find("world")
            for index in range(2):
                camera = scene.find(f"model[@name='annotation_camera_{index}']")
                camera.remove(camera.find("pose"))
            normalized.append(ET.tostring(tree.getroot()))
        self.assertEqual(normalized[0], normalized[1])

    def test_target_collision_matches_scaled_body(self):
        tree = self.build_controls()[0][0]
        for target in ("probe_red", "probe_blue"):
            link = tree.getroot().find(f"world/model[@name='{target}']/link")
            body = link.find("visual[@name='body']/geometry")
            collision = link.find("collision/geometry")
            self.assertEqual(ET.tostring(body), ET.tostring(collision))

    def test_unknown_case_rejected_before_resource_reads(self):
        with self.assertRaisesRegex(ValueError, "viewpoint"):
            static_world(self.root / "missing", self.root / "missing", "unknown")

    def test_aircraft_scene_keeps_background_and_replaces_static_cameras(self):
        static, _, resources = self.build_controls()[1]
        with patch("benchmark.px4_gazebo.model_handoff.warehouse_scene.static_world",
                   return_value=(deepcopy(static), [], resources)):
            tree, contract = aircraft_world(self.root, self.root / "cameras")
        scene = tree.getroot().find("world")
        self.assertEqual(scene.get("name"), RUNTIME_WORLD)
        self.assertEqual(scene.findtext("spherical_coordinates/world_frame_orientation"), "ENU")
        self.assertEqual(scene.findtext("spherical_coordinates/surface_model"), "EARTH_WGS84")
        self.assertIsNotNone(scene.find("magnetic_field"))
        self.assertFalse(scene.findall("model[@name='annotation_camera_0']"))
        self.assertEqual(len(scene.findall("include")), 2)
        self.assertEqual(len(scene.findall(".//collision")), len(static.getroot().findall("world/.//collision")))
        self.assertIsNone(contract["role_assignment"])
        self.assertFalse(contract["flight_admitted"])
        for index, include in enumerate(scene.findall("include")):
            self.assertEqual(include.findtext("name"), f"annotation_uav_{index}")
            self.assertEqual(tuple(map(float, include.findtext("pose").split())), AIRCRAFT_POSES[index])
        names = {p.get("name") for p in scene.findall("plugin")}
        for name in ("Contact", "Imu", "NavSat", "AirPressure", "Magnetometer", "UserCommands"):
            self.assertIn(f"gz::sim::systems::{name}", names)

    def test_export_private_contract_fingerprints_world_and_does_not_overwrite(self):
        static, _, resources = self.build_controls()[1]
        with patch("benchmark.px4_gazebo.model_handoff.warehouse_startup.aircraft_world",
                   return_value=(static, {**resources, "world_name": RUNTIME_WORLD})):
            output = self.root / "export"
            result = export(self.root, self.root, output)
            self.assertEqual(result["world_sha256"], sha256(output / "world.sdf"))
            self.assertFalse(result["flight_admitted"])
            self.assertEqual(result["motion_commands_sent"], 0)
            self.assertTrue((output / "scene_contract.private.json").is_file())
            with self.assertRaises(FileExistsError):
                export(self.root, self.root, output)


if __name__ == "__main__":
    unittest.main()
