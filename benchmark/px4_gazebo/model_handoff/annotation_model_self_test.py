"""Synthetic opt-in SDF generation checks; no flight readiness claim."""

import unittest
import xml.etree.ElementTree as ET

from .annotation_models import CAMERA_MODEL, LABEL_SENSOR, paired_camera

SOURCE = """<sdf version='1.9'><model name='mono_cam'><link name='camera_link'>
<sensor name='camera' type='camera'><pose>0 0 0 0 0 0</pose><camera>
<horizontal_fov>1.74</horizontal_fov><image><width>1280</width><height>960</height></image>
<clip><near>0.1</near><far>3000</far></clip></camera><always_on>1</always_on>
<update_rate>30</update_rate><visualize>true</visualize></sensor>
</link></model></sdf>"""


class AnnotationModelTests(unittest.TestCase):
    def test_source_unchanged_and_sensor_parameters_matched(self):
        source = ET.fromstring(SOURCE)
        before = ET.tostring(source)
        result = paired_camera(source)
        self.assertEqual(ET.tostring(source), before)
        self.assertEqual(result.find("model").get("name"), CAMERA_MODEL)
        rgb, labels = result.findall("model/link/sensor")
        self.assertEqual(ET.tostring(rgb), ET.tostring(source.find("model/link/sensor")))
        self.assertEqual(labels.get("name"), LABEL_SENSOR)
        self.assertEqual(labels.get("type"), "segmentation")
        for field in ("pose", "camera/image", "camera/clip", "camera/horizontal_fov", "update_rate"):
            self.assertEqual(ET.tostring(rgb.find(field)), ET.tostring(labels.find(field)))
        self.assertEqual(labels.findtext("camera/segmentation_type"), "instance")
        self.assertEqual(labels.findtext("visualize"), "false")

    def test_unknown_camera_structure_rejected(self):
        for field, value in (("model", "wrong"), ("model/link/sensor", "depth_camera")):
            root = ET.fromstring(SOURCE)
            root.find(field).set("name" if field == "model" else "type", value)
            with self.assertRaises(ValueError):
                paired_camera(root)

    def test_topic_distortion_and_noise_rejected(self):
        for parent, tag in (("model/link/sensor", "topic"),
                            ("model/link/sensor/camera", "noise"),
                            ("model/link/sensor/camera", "distortion")):
            root = ET.fromstring(SOURCE)
            ET.SubElement(root.find(parent), tag).text = "/shared"
            with self.assertRaises(ValueError):
                paired_camera(root)

    def test_dimensions_and_rate_rejected(self):
        for field in ("camera/image/width", "camera/image/height", "update_rate"):
            root = ET.fromstring(SOURCE)
            root.find("model/link/sensor/" + field).text = "0"
            with self.assertRaises(ValueError):
                paired_camera(root)


if __name__ == "__main__":
    unittest.main()
