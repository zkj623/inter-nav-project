import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel, origin_transform


class OriginTransformTests(unittest.TestCase):
    def test_missing_origin_is_identity(self):
        np.testing.assert_allclose(origin_transform(None), np.eye(4))


@unittest.skipUnless(importlib.util.find_spec('fcl'), 'requires the isolated python-fcl dependency')
class CollisionEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        urdf = Path(self.directory.name) / 'test.urdf'
        urdf.write_text('''<robot name="collision_fixture">
          <link name="base_link">
            <visual><geometry><box size="1 1 1"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="tool_link">
            <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
          </link>
          <joint name="slide" type="prismatic">
            <parent link="base_link"/><child link="tool_link"/>
            <origin xyz="0 0 1"/><axis xyz="1 0 0"/>
          </joint>
        </robot>''')
        self.model = RobotCollisionModel(urdf)
        self.transforms = self.model.forward({'slide': 0.2})

    def tearDown(self):
        self.directory.cleanup()

    def test_forward_includes_prismatic_motion_and_base_pose(self):
        base = np.eye(4)
        base[:3, 3] = (2.0, 3.0, 4.0)
        transforms = self.model.forward({'slide': 0.2}, base)
        np.testing.assert_allclose(transforms['tool_link'][:3, 3], (2.2, 3.0, 5.0))

    def test_visual_shell_not_undersized_proxy_defines_clearance(self):
        obstacle = np.eye(4)
        obstacle[:3, 3] = (0.8, 0.0, 0.0)
        distances = self.model.box_clearances(self.transforms, obstacle, (0.2, 0.2, 0.2))
        self.assertAlmostEqual(distances['base_link'], 0.2, places=5)

    def test_visual_penetration_is_detected_despite_small_physics_proxy(self):
        obstacle = np.eye(4)
        obstacle[:3, 3] = (0.45, 0.0, 0.0)
        distances = self.model.box_clearances(self.transforms, obstacle, (0.2, 0.2, 0.2))
        self.assertLess(distances['base_link'], 0.0)

    def test_contained_box_is_not_mistaken_for_surface_clearance(self):
        distances = self.model.box_clearances(self.transforms, np.eye(4), (0.1, 0.1, 0.1))
        self.assertLess(distances['base_link'], 0.0)

    def test_direct_joint_neighbors_are_excluded_from_self_surface_test(self):
        self.assertNotIn(('base_link', 'tool_link'), self.model.self_pairs)


if __name__ == '__main__':
    unittest.main()
