from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask


class LoweringPoseTests(unittest.TestCase):
    def test_multiple_lower_segments_share_continuous_object_poses(self):
        environment = SimpleNamespace(task=ClearanceTask())
        center = np.array((0.75, 1.25, 0.22))
        current = np.eye(4)
        current[:3, :3] = Rotation.from_euler('y', 0.08).as_matrix()
        current[:3, 3] = center + current[:3, :3] @ np.array((0, 0, 0.14))
        poses = []

        def record(environment, joints, base, box):
            poses.append(box.copy())
            return {'minimum_noncontact_clearance_m': 0.02, 'minimum_wall_clearance_m': 0.10}

        with patch('grutopia_extension.interaction_clearance.primitives.check_configuration', side_effect=record):
            ArmPlanPolicy(environment).certify('place', np.zeros(19), [np.ones(19), np.zeros(19)],
                                               ['lower', 'lower'], np.eye(4), center, 0.0, current)
        np.testing.assert_allclose(poses[100], poses[101], atol=1e-12)
        self.assertGreater(poses[100][2, 3], 0.20)
        np.testing.assert_allclose(poses[-1][:3, 3], (*current[:2, 3], 0.1401), atol=1e-12)

    def test_lowering_starts_at_measured_tilted_pose_and_finishes_grounded(self):
        task = ClearanceTask(object_kind='bin', dimensions=(0.34, 0.34, 0.34))
        environment = SimpleNamespace(task=task)
        center = np.array((0.75, 1.25, 0.22))
        current = np.eye(4)
        current[:3, :3] = Rotation.from_euler('y', 0.08).as_matrix()
        current[:3, 3] = center + current[:3, :3] @ np.array((0, 0, 0.17))
        original = current.copy()
        poses = []

        def record(environment, joints, base, box):
            poses.append(box.copy())
            return {'minimum_noncontact_clearance_m': 0.02, 'minimum_wall_clearance_m': 0.10}

        with patch('grutopia_extension.interaction_clearance.primitives.check_configuration', side_effect=record):
            ArmPlanPolicy(environment).certify('place', np.zeros(19), [np.ones(19), np.zeros(19)],
                                               ['lower', 'release'], np.eye(4), center, 0.0, current)
        np.testing.assert_allclose(current, original)
        np.testing.assert_allclose(poses[0], original, atol=1e-12)
        np.testing.assert_allclose(poses[100][:3, 3], (*current[:2, 3], 0.1701))
        np.testing.assert_allclose(poses[100][:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(poses[101], poses[100], atol=1e-12)
        self.assertLess(np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3]), 0.003)


if __name__ == '__main__':
    unittest.main()
