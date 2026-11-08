from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import ArmExecutePolicy, ArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask


class CartesianLiftTests(unittest.TestCase):
    def test_all_household_types_replan_using_measured_lift_progress(self):
        for kind in ('box', 'bin', 'stool'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                position = np.zeros(3)
                task = ClearanceTask(object_kind=kind)
                environment = SimpleNamespace(task=task, box=SimpleNamespace(
                    dimensions=task.dimensions, squeeze=0.003, grasp_height=0.25),
                    finger_force_vectors=np.array(((0, 10, 0), (0, 10, 0), (0, -10, 0), (0, -10, 0))),
                    object=SimpleNamespace(get_world_pose=lambda: (position.copy(), np.array((1., 0., 0., 0.)))),
                    base=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1., 0., 0., 0.)))),
                    rotation=lambda quaternion: np.eye(3), command=np.zeros(19), control_indices=np.arange(19),
                    solve_both=Mock(return_value=np.zeros(15)), output=Path(directory), elapsed_steps=0,
                    recorder=SimpleNamespace(event=Mock()))
                policy = ArmExecutePolicy(environment, 'grasp')

                def motion(target, duration, phase):
                    if phase == 'lift':
                        position[2] += 0.025  # Measured lift undershoots each nominal 3.7 cm segment.
                    environment.elapsed_steps += 1

                plan = {'targets': [np.zeros(19)] * 8, 'phases': ['approach', 'squeeze'] + ['lift'] * 6,
                        'durations': [5, 4] + [0.8] * 6}
                audit = [{'minimum_noncontact_clearance_m': 0.05, 'minimum_wall_clearance_m': 0.08}]
                with patch.object(policy, 'interpolate', side_effect=motion), \
                        patch.object(ArmPlanPolicy, 'certify', return_value=audit) as certify:
                    policy.grasp_motion(plan)
                calls = environment.solve_both.call_args_list
                self.assertEqual(len(calls), 6)
                self.assertEqual(certify.call_count, 6)
                self.assertAlmostEqual(calls[0].args[2], 0.25 + 0.22 / 6)
                self.assertAlmostEqual(calls[-1].args[2], 0.25 + 0.22 - 5 * 0.025)

    def test_lift_segments_have_continuous_object_prediction(self):
        environment = SimpleNamespace(task=ClearanceTask())
        box = np.eye(4)
        box[:3, :3] = Rotation.from_euler('y', 0.04).as_matrix()
        box[:3, 3] = (0.75, 0.0, 0.14)
        poses = []

        def record(environment, joints, base, pose):
            poses.append(pose.copy())
            return {'minimum_noncontact_clearance_m': 0.03, 'minimum_wall_clearance_m': 0.10}

        with patch('grutopia_extension.interaction_clearance.primitives.check_configuration', side_effect=record):
            ArmPlanPolicy(environment).certify('grasp', np.zeros(19), [np.ones(19), np.zeros(19)],
                ['lift', 'lift'], np.eye(4), np.array((0.75, 0.0, 0.0)), 0.0, box)
        np.testing.assert_allclose(poses[0], box)
        np.testing.assert_allclose(poses[100], poses[101])
        np.testing.assert_allclose(poses[-1][:3, 3], (0.75, 0.0, 0.36))
        np.testing.assert_allclose(poses[-1][:3, :3], box[:3, :3])

    def test_missing_contact_stops_before_any_lift_motion(self):
        environment = SimpleNamespace(finger_force_vectors=np.zeros((4, 3)),
            object=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0)))))
        policy = ArmExecutePolicy(environment, 'grasp')
        plan = {'targets': [np.zeros(19)] * 3, 'phases': ['approach', 'squeeze', 'lift'], 'durations': [5, 4, 4]}
        with patch.object(policy, 'interpolate') as motion:
            with self.assertRaisesRegex(RuntimeError, 'GRASP_CONTACT_MISSING_BEFORE_LIFT'):
                policy.grasp_motion(plan)
        self.assertEqual([call.args[2] for call in motion.call_args_list], ['approach', 'squeeze'])


if __name__ == '__main__':
    unittest.main()
