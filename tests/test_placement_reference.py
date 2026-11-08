from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy, BaseMovePolicy, grasp_standoff_m
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox


class PlacementReferenceTests(unittest.TestCase):
    def test_grasp_standoff_reserves_space_for_extent_and_stool_settling(self):
        self.assertEqual(grasp_standoff_m(ClearanceTask()), 0.65)
        self.assertAlmostEqual(grasp_standoff_m(ClearanceTask(dimensions=(0.48, 0.58, 0.34))), 0.66)
        self.assertEqual(grasp_standoff_m(ClearanceTask(object_kind='stool')), 0.72)
        self.assertLessEqual(grasp_standoff_m(ClearanceTask(object_kind='stool', grasp_distance_m=0.72)), 0.72)

    def environment(self, horizontal_offset=0.0):
        task = ClearanceTask(object_kind='stool', place_xy=(0.78, 1.25))
        matrix = Rotation.from_euler('y', 0.44).as_matrix()
        quaternion = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
        center = np.array((0.78 + horizontal_offset, 1.25, 0.37))
        origin = center - matrix @ np.array((0, 0, 0.14))
        return SimpleNamespace(task=task, box=BimanualBox(), interaction_state='holding', selected_place_xy=np.array(task.place_xy),
            base=SimpleNamespace(get_world_pose=lambda: (np.array((0.17, 1.25, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))),
            object=SimpleNamespace(get_world_pose=lambda: (origin.copy(), quaternion.copy())),
            rotation=lambda value: Rotation.from_quat(np.asarray(value)[[1, 2, 3, 0]]).as_matrix())

    def test_tilt_does_not_create_a_false_base_translation(self):
        environment = self.environment()
        self.assertGreater(abs(environment.object.get_world_pose()[0][0] - 0.78), 0.055)
        np.testing.assert_allclose(BaseMovePolicy(environment).target('placement'), (0.17, 1.25, 0.0), atol=1e-12)

    def test_actual_center_displacement_still_requires_correction(self):
        environment = self.environment(0.10)
        np.testing.assert_allclose(BaseMovePolicy(environment).target('placement'), (0.07, 1.25, 0.0), atol=1e-12)
        outcome = ArmPlanPolicy(environment).execute(environment.task, mode='place')
        self.assertEqual(outcome.reason, 'OBJECT_NOT_AT_PLACEMENT')


if __name__ == '__main__':
    unittest.main()
