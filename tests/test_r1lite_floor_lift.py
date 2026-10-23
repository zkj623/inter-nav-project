import unittest
from pathlib import Path
import tempfile
import json

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.floor_lift import (
    LIFT_ASSETS, LiftSuccessTracker, UrdfArmKinematics, lowest_box_point, policy_observation,
)


class FloorLiftTests(unittest.TestCase):
    def test_assets_have_graspable_handles(self):
        for asset in LIFT_ASSETS.values():
            self.assertGreater(asset.mass, 0)
            self.assertGreater(asset.handle_height, asset.dimensions[2])
            self.assertLess(asset.handle_width, 0.09)

    def test_continuous_hold_is_required(self):
        tracker = LiftSuccessTracker(0.0, required_lift=0.15, hold_seconds=0.3)
        self.assertFalse(tracker.update(0.2, 0.1, True))
        self.assertFalse(tracker.update(0.2, 0.1, True))
        self.assertFalse(tracker.update(0.1, 0.1, True))
        self.assertFalse(tracker.update(0.2, 0.1, True))
        self.assertFalse(tracker.update(0.2, 0.1, True))
        self.assertTrue(tracker.update(0.2, 0.1, True))

    def test_grasp_and_health_are_required(self):
        for grasped, healthy in ((False, True), (True, False)):
            tracker = LiftSuccessTracker(0.0, hold_seconds=0.1)
            self.assertFalse(tracker.update(0.3, 0.2, grasped, healthy))

    def test_nonfinite_state_cannot_succeed(self):
        tracker = LiftSuccessTracker(0.0)
        self.assertFalse(tracker.update(float('nan'), 3.0, True))

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            LiftSuccessTracker(float('nan'))
        with self.assertRaises(ValueError):
            LiftSuccessTracker(0.0, required_lift=0)
        with self.assertRaises(ValueError):
            LiftSuccessTracker(0.0).update(0.2, float('nan'), True)

    def test_bounded_ik_matches_forward_kinematics(self):
        urdf = '<robot name="test"><link name="base_link"/><link name="right_gripper_link"/>'
        urdf += '<joint name="arm" type="revolute"><parent link="base_link"/>'
        urdf += '<child link="right_gripper_link"/><axis xyz="0 0 1"/>'
        urdf += '<limit lower="-3.14" upper="3.14" effort="1" velocity="1"/></joint></robot>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.urdf'
            path.write_text(urdf)
            kinematics = UrdfArmKinematics(path, tool_offset=(1, 0, 0))
            orientation = Rotation.from_euler('z', 0.5).as_matrix()
            positions = kinematics.solve((np.cos(0.5), np.sin(0.5), 0), orientation, (0.2,))
            self.assertAlmostEqual(positions[0], 0.5, places=4)
            with self.assertRaises(RuntimeError):
                kinematics.solve((2, 0, 0), np.eye(3), (0.0,))

    def test_bottom_tracks_rotated_geometry(self):
        self.assertAlmostEqual(lowest_box_point((0, 0, 0.4), np.eye(3), (0.2, 0.3, 0.4)), 0.2)
        self.assertAlmostEqual(lowest_box_point(
            (0, 0, 0.4), Rotation.from_euler('x', 90, degrees=True).as_matrix(), (0.2, 0.3, 0.4)
        ), 0.25)

    def test_observation_schema(self):
        observation = policy_observation(np.zeros(11), (0.6, -0.3, 0.1), 2, 0.5, (0.2, 0.2, 0.2))
        self.assertEqual(observation.shape, (23,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation[-6:-1].tolist(), [0, 0, 1, 0, 0])

    def test_failed_demonstrations_are_not_training_data(self):
        from grutopia_extension.manipulation.lift_policy import train_policy

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'result.json').write_text(json.dumps({'success': False}))
            with self.assertRaises(ValueError):
                train_policy([directory / 'demonstration.npz'], directory / 'policy.pt', epochs=1)

    def test_single_arm_trainer_rejects_bimanual_schema(self):
        from grutopia_extension.manipulation.lift_policy import train_policy

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'result.json').write_text(json.dumps({'success': True}))
            np.savez(directory / 'demonstration.npz', observations=np.zeros((40, 30)), actions=np.zeros((40, 19)))
            with self.assertRaises(ValueError):
                train_policy([directory / 'demonstration.npz'], directory / 'policy.pt', epochs=1)

    def test_policy_masks_unstable_feedback_features(self):
        import torch
        from grutopia_extension.manipulation.lift_policy import load_policy, make_network

        mask = torch.ones(23)
        mask[:11] = 0
        mask[13] = 0
        checkpoint = {
            'state_dict': make_network(23, 11).state_dict(), 'feature_mask': mask,
            'observation_mean': torch.zeros(23), 'observation_scale': torch.ones(23),
            'action_mean': torch.zeros(11), 'action_scale': torch.ones(11),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            torch.save(checkpoint, path)
            policy = load_policy(path)
            first = np.zeros(23, dtype=np.float32)
            second = first.copy()
            second[:11] = 10
            second[13] = 10
            np.testing.assert_array_equal(policy(first), policy(second))


if __name__ == '__main__':
    unittest.main()
