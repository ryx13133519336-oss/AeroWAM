"""SubT conversion/provenance/geometry regressions, without Gazebo or ROS."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import subt_urban_scene as s
from .subt_urban import export
from .subt_urban_geometry import collada_triangles, segment_hits

TEXTURE = "https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Urban Platform/tip/files/materials/textures/test.jpg"
MODEL = f'''<sdf version="1.6"><model name="source"><static>true</static><link name="walls">
<pose>0 0 -1.7 0 0 0</pose><inertial><mass>1</mass></inertial>
<collision name="wall"><geometry><mesh><uri>meshes/model.dae</uri><scale>1 1 1</scale></mesh></geometry></collision>
<visual name="wall"><geometry><mesh><uri>meshes/model.dae</uri><submesh><name>Wall</name><center>false</center></submesh></mesh></geometry>
<material><script><uri>file://unneeded</uri></script><pbr><metal><albedo_map>{TEXTURE}</albedo_map></metal></pbr></material></visual>
<plugin filename="libignition-gazebo-performer-detector-system.so" name="ignition::gazebo::systems::PerformerDetector"/>
</link></model></sdf>'''
DAE = f'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_images><image><init_from>{TEXTURE}</init_from></image></library_images></COLLADA>'''


class SubTTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.package = self.root / "package"
        self.assets = deepcopy(s.ASSETS)
        for key, asset in self.assets.items():
            path = self.source / f"{key}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in {"model.sdf": MODEL, "meshes/model.dae": DAE,
                                   "model.config": "authors", "metadata.pbtxt": "license",
                                   "materials/textures/test.jpg": b"texture"}.items():
                    archive.writestr(name, data)
            asset["sha256"] = s.sha256(path)
            metadata = {"owner": "OpenRobotics", "name": asset["name"], "version": asset["version"],
                        "license_name": "Creative Commons Attribution 4.0 International"}
            (self.source / f"{key}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.patch = patch.object(s, "ASSETS", self.assets)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def prepared(self):
        with patch.object(s, "verify", return_value={}):
            manifest = s.prepare(self.source, self.package)
        self.pin_table(manifest)
        return manifest

    def pin_table(self, manifest):
        files = {k: v for k, v in manifest["files"].items() if not k.endswith("/fuel_metadata.json")}
        table = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        p = patch.object(s, "RESOURCE_TABLE_SHA256", table)
        p.start()
        self.addCleanup(p.stop)

    def test_prepare_offline_collada_and_attribution(self):
        self.prepared()
        s.verify(self.package)
        mesh = self.package / "urban_2_story/meshes/model.dae"
        raw = mesh.read_text(encoding="utf-8")
        self.assertIn("<COLLADA", raw)
        self.assertNotIn("ns0:", raw)
        self.assertNotIn("https:", raw)
        image = ET.parse(mesh).find(".//{*}init_from")
        self.assertTrue((mesh.parent / image.text).resolve().is_file())
        self.assertIn("Creative Commons", (self.package / "ATTRIBUTION.txt").read_text())
        with self.assertRaises(FileExistsError):
            s.prepare(self.source, self.package)

    def test_bad_archive_rejected_before_creating_package(self):
        (self.source / "urban_2_story.zip").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "archive mismatch"):
            s.prepare(self.source, self.package)
        self.assertFalse(self.package.exists())

    def test_rehashed_package_tampering_rejected(self):
        manifest = self.prepared()
        target = self.package / "urban_2_story/model.sdf"
        target.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            s.verify(self.package)
        manifest["files"]["urban_2_story/model.sdf"] = s.sha256(target)
        (self.package / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "resource table"):
            s.verify(self.package)

    def test_all_asset_versions_remain_pinned(self):
        self.prepared()
        path = self.package / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["assets"]["urban_2_story"]["version"] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance"):
            s.verify(self.package)

    def test_conversion_preserves_collision_units_submesh_and_link_pose(self):
        self.prepared()
        before = ET.fromstring(MODEL).find("model/link/collision")
        before.find("geometry/mesh/uri").text = (self.package / "urban_2_story/meshes/model.dae").as_uri()
        model = s.model_copy(self.package, "urban_2_story", "building")
        self.assertEqual(ET.tostring(model.find("link/collision")), ET.tostring(before))
        self.assertEqual(model.findtext("link/pose"), "0 0 -1.7 0 0 0")
        self.assertEqual(model.findtext("link/visual/geometry/mesh/submesh/name"), "Wall")
        self.assertEqual(model.findtext("link/visual/geometry/mesh/submesh/center"), "false")
        self.assertFalse(model.findall(".//inertial"))
        self.assertFalse(model.findall(".//material/script"))
        self.assertEqual(len(model.findall(".//plugin")), 1)
        self.assertEqual(model.findtext("plugin/label"), "100")
        self.assertTrue(model.findtext("link/visual/material/pbr/metal/albedo_map").startswith("file:"))

    def test_arbitrary_plugins_and_includes_rejected(self):
        self.prepared()
        path = self.package / "urban_2_story/model.sdf"
        for suffix, error in (("<plugin filename='fake-performer-detector' name='evil'/>", "plugin"),
                              ("<include><uri>http://example.com</uri></include>", "nested")):
            path.write_text(MODEL.replace("</model>", suffix + "</model>"))
            with self.assertRaisesRegex(ValueError, error):
                s.model_copy(self.package, "urban_2_story", "building")

    def test_texture_hosts_versions_traversal_and_missing_rejected(self):
        self.prepared()
        self.assertTrue(s.texture_resource(self.package, TEXTURE).is_file())
        for uri in (TEXTURE.replace("https:", "http:"), TEXTURE.replace("tip/", "99/"),
                    TEXTURE.replace("fuel.ignitionrobotics.org", "evil.com"), TEXTURE + "?remote=1",
                    TEXTURE.replace("test.jpg", "../../../outside.jpg"), "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                s.texture_resource(self.package, uri)

    def test_wrong_license_fails_closed(self):
        path = self.source / "urban_2_story.metadata.json"
        path.write_text(json.dumps({"name": "Urban 2 Story", "version": 14, "license_name": "unknown"}))
        with self.assertRaisesRegex(ValueError, "identity or license"):
            s.prepare(self.source, self.package)

    def controls(self, aircraft=False):
        self.prepared()
        with patch.object(s, "verify_cameras", return_value={}):
            return [s.scene(self.package, self.root / "cameras", case=case, aircraft=aircraft) for case in s.CAMERAS]

    def test_controls_identical_except_camera_poses_and_roles_unassigned(self):
        normalized = []
        for tree, contract in self.controls():
            self.assertIsNone(contract["role_assignment"])
            self.assertFalse(contract["flight_admitted"])
            self.assertFalse(contract["policy_evaluated"])
            self.assertEqual(len(contract["camera_pairs"]), 2)
            for index in range(2):
                model = tree.find(f"world/model[@name='annotation_camera_{index}']")
                model.remove(model.find("pose"))
            normalized.append(ET.tostring(tree.getroot()))
        self.assertEqual(len(set(normalized)), 1)

    def test_aircraft_export_no_flight_publishers_and_gz_plugins_present(self):
        tree, contract = self.controls(aircraft=True)[0]
        world = tree.find("world")
        self.assertFalse(world.findall("model[@name='annotation_camera_0']"))
        self.assertEqual(len(world.findall("include")), 2)
        self.assertEqual(world.findtext("spherical_coordinates/world_frame_orientation"), "ENU")
        self.assertEqual(world.findtext("physics/max_step_size"), ".004")
        self.assertIsNone(contract["role_assignment"])
        self.assertFalse(contract["flight_admitted"])
        self.assertTrue(world.findall("gui/plugin[@filename='MinimalScene']"))

    def test_unknown_case_and_existing_export_rejected(self):
        with self.assertRaisesRegex(ValueError, "diagnostic view"):
            s.scene("missing", "missing", case="wrong")
        tree, contract = self.controls()[0]
        output = self.root / "export"
        with patch("benchmark.px4_gazebo.model_handoff.subt_urban.scene", return_value=(tree, contract)):
            export(self.package, self.root / "cameras", output)
            with self.assertRaises(FileExistsError):
                export(self.package, self.root / "cameras", output)


class RayTests(unittest.TestCase):
    def test_two_sided_segment_endpoints_and_no_hit(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        self.assertTrue(np.allclose(segment_hits(triangle, [.2, .2, 1], [.2, .2, -1]), [.5]))
        self.assertTrue(np.allclose(segment_hits(triangle, [.2, .2, -1], [.2, .2, 1]), [.5]))
        self.assertEqual(len(segment_hits(triangle, [2, 2, 1], [2, 2, -1])), 0)
        self.assertEqual(len(segment_hits(triangle, [.2, .2, 0], [.2, .2, -1])), 0)

    def test_collada_centimeters_matrix_and_index_stride(self):
        root = ET.fromstring('''<COLLADA><asset><unit meter="0.01"/><up_axis>Z_UP</up_axis></asset>
<library_geometries><geometry id="g"><mesh><source id="p"><float_array>0 0 0 100 0 0 0 100 0</float_array>
<technique_common><accessor stride="3"/></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
<triangles><input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#unused" offset="1"/>
<p>0 0 1 0 2 0</p></triangles></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node><matrix>1 0 0 200 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
<instance_geometry url="#g"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>''')
        self.assertTrue(np.allclose(collada_triangles(root), [[[2, 0, 0], [3, 0, 0], [2, 1, 0]]]))
        root.find("asset/up_axis").text = "Y_UP"
        with self.assertRaisesRegex(ValueError, "Z-up"):
            collada_triangles(root)


if __name__ == "__main__":
    unittest.main()
