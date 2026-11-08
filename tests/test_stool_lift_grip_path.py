from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import ArmExecutePolicy, ArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask


class StoolLiftGripTests(unittest.TestCase):
    def planner(self, half_width):
        def forward(joints):
            width = half_width(joints) if callable(half_width) else half_width
            tools = [np.eye(4), np.eye(4)]
            tools[0][1, 3], tools[1][1, 3] = width, -width
            return tools
        task = ClearanceTask(object_kind='stool', dimensions=(.44, .52, .36), mass_kg=1.8)
        return ArmPlanPolicy(SimpleNamespace(task=task, box=SimpleNamespace(squeeze=.003),
                                            bimanual=SimpleNamespace(forward=forward)))

    def test_lift_path_rejects_middle_widening_with_accurate_endpoints(self):
        planner = self.planner(lambda q: .257 + .006 * np.sin(np.pi * q[0]))
        initial, target = np.zeros(19), np.ones(19)
        planner.check_lift_grip(initial, np.eye(4), np.eye(4))
        planner.check_lift_grip(target, np.eye(4), np.eye(4))
        audit = {'minimum_noncontact_clearance_m': .05, 'minimum_wall_clearance_m': .1}
        with patch('grutopia_extension.interaction_clearance.primitives.check_configuration', return_value=audit):
            with self.assertRaisesRegex(RuntimeError, 'LIFT_GRIP_PATH_INACCURATE'):
                planner.certify('grasp', initial, [target], ['lift'], np.eye(4), np.zeros(3), 0, np.eye(4))

    def test_contact_span_is_invariant_to_rigid_base_object_rotation(self):
        for euler in ((0, 0, .7), (.03, -.06, -1.2)):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', euler).as_matrix()
            transform[:3, 3] = (2, 3, .4)
            detail = self.planner(.2574).check_lift_grip(np.zeros(19), transform, transform)
            self.assertAlmostEqual(detail['grip_span_error_m'], .0004)

    def test_widening_excess_inward_preload_and_nonfinite_are_rejected(self):
        for width in (.260, .2555, np.nan):
            with self.subTest(width=width), self.assertRaisesRegex(RuntimeError, 'LIFT_GRIP_PATH_INACCURATE'):
                self.planner(width).check_lift_grip(np.zeros(19), np.eye(4), np.eye(4))

    def test_each_lift_control_checks_contact_and_stops_on_first_loss(self):
        environment = SimpleNamespace(task=SimpleNamespace(object_kind='stool'), control_dt=.02,
            command=np.zeros(19), control_indices=np.arange(19),
            finger_force_vectors=np.array(((0, 10., 0), (0, 10., 0), (0, -10., 0), (0, -10., 0))),
            object=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1., 0., 0., 0.)))))
        executed = []

        def advance(joints, phase, progress):
            executed.append(joints)
            if len(executed) == 3:
                environment.finger_force_vectors[2:] = 0

        environment.advance = advance
        policy = ArmExecutePolicy(environment, 'grasp')
        with patch.object(policy, 'phase_end') as ended:
            with self.assertRaisesRegex(RuntimeError, 'GRASP_LOST_DURING_LIFT'):
                policy.interpolate(np.ones(19), .8, 'lift')
            ended.assert_not_called()
        self.assertEqual(len(executed), 3)

    def test_successful_contact_preserves_full_motion_and_endpoint(self):
        environment = SimpleNamespace(task=SimpleNamespace(object_kind='stool'), control_dt=.02,
            command=np.zeros(19), control_indices=np.arange(19),
            finger_force_vectors=np.array(((0, 10., 0), (0, 10., 0), (0, -10., 0), (0, -10., 0))),
            object=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1., 0., 0., 0.)))), advance=Mock())
        policy = ArmExecutePolicy(environment, 'grasp')
        with patch.object(policy, 'phase_end') as ended:
            policy.interpolate(np.ones(19), .8, 'lift')
            ended.assert_called_once_with('lift')
        self.assertEqual(environment.advance.call_count, 40)
        np.testing.assert_allclose(environment.advance.call_args.args[0], np.ones(19))


if __name__ == '__main__':
    unittest.main()
