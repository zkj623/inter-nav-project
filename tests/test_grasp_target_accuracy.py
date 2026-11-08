from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.motion_geometry import grasp_frames
from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask, yaw_rotation
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox


class GraspTargetAccuracyTests(unittest.TestCase):
    def environment(self, errors, *, base_yaw=0.0, object_yaw=0.0):
        center = np.array((0.72, 0.0, 0.0001))
        base = np.array((0.0, 0.0, 0.0))
        rotation = yaw_rotation(base_yaw)
        task = ClearanceTask(object_kind='stool', dimensions=(.34, .42, .28), mass_kg=1.1,
                             base_obstacle_clearance_m=.08, arm_obstacle_clearance_m=.08,
                             execution_obstacle_clearance_m=.03)
        box = BimanualBox(task.dimensions, task.mass_kg, .265, .003, 'stool')
        positions = grasp_frames(center, object_yaw, .207, .265)[:2]
        tools = []
        for position, error in zip(positions, errors):
            tool = np.eye(4)
            tool[:3, 3] = rotation.T @ (position + yaw_rotation(object_yaw) @ error - base)
            tools.append(tool)
        env = SimpleNamespace(task=task, box=box,
            bimanual=SimpleNamespace(forward=Mock(return_value=tools)),
            base=SimpleNamespace(get_world_pose=lambda: (base, Rotation.from_euler('z', base_yaw).as_quat()[[3, 0, 1, 2]])),
            object=SimpleNamespace(get_world_pose=lambda: (center, Rotation.from_euler('z', object_yaw).as_quat()[[3, 0, 1, 2]])),
            rotation=lambda q: Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix())
        return env, center

    def test_recorded_miss_is_rejected_despite_old_8mm_ik_acceptance(self):
        errors = (np.array((-.000647, .005926, .004934)), np.array((-.000634, -.006390, .004746)))
        self.assertTrue(all(np.linalg.norm(e) < .008 for e in errors))
        env, center = self.environment(errors)
        with self.assertRaisesRegex(RuntimeError, 'GRASP_TARGET_INACCURATE'):
            ArmPlanPolicy(env).check_squeeze_target(np.zeros(19), center)

    def test_both_hands_need_normal_accuracy_in_rotated_frames(self):
        for errors, accepted in [(((.006, .0004, .004), (-.006, -.0004, .004)), True),
                                 (((0, 0, 0), (0, -.0015, 0)), False),
                                 (((0, -.0015, 0), (0, 0, 0)), False)]:
            with self.subTest(errors=errors):
                env, center = self.environment(errors, base_yaw=.6, object_yaw=-.3)
                if accepted:
                    detail = ArmPlanPolicy(env).check_squeeze_target(np.zeros(19), center)
                    np.testing.assert_allclose(detail['normal_errors_m'], (.0004, .0004), atol=1e-12)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'GRASP_TARGET_INACCURATE'):
                        ArmPlanPolicy(env).check_squeeze_target(np.zeros(19), center)

    def test_nonfinite_target_is_rejected(self):
        env, center = self.environment(((0, np.nan, 0), (0, 0, 0)))
        with self.assertRaisesRegex(RuntimeError, 'GRASP_TARGET_INACCURATE'):
            ArmPlanPolicy(env).check_squeeze_target(np.zeros(19), center)

    def test_inaccurate_grip_selects_next_candidate_before_execution(self):
        env, center = self.environment(((0, 0, 0), (0, 0, 0)))
        exact_tools = env.bimanual.forward(np.zeros(15))
        bad_tools = [tool.copy() for tool in exact_tools]
        bad_tools[0][1, 3] += .006
        bad_tools[1][1, 3] -= .006
        env.bimanual.forward = lambda q: bad_tools if q[0] == 1 else exact_tools

        def solve(center, width, height, seed, object_rotation=None, preserve_contact_normal=False):
            if width > .21:
                if width > .33:
                    raise RuntimeError('unreachable approach')
                marker = 1 if width > .31 else 2
            else:
                marker = seed[0]
            q = np.zeros(15); q[0] = marker
            return q

        with tempfile.TemporaryDirectory() as directory:
            env.solve_both = solve
            env.interaction_state = 'ready'
            env.arm_plan = None
            env.initial_control = np.zeros(19)
            env.robot = SimpleNamespace(get_joint_positions=lambda: np.zeros(19))
            env.control_indices = np.arange(19)
            env.elapsed_steps = 0
            env.output = Path(directory)
            env.recorder = SimpleNamespace(event=Mock())
            policy = ArmPlanPolicy(env)
            audit = [{'minimum_noncontact_clearance_m': .048, 'minimum_wall_clearance_m': .24}]
            with patch.object(policy, 'certify', return_value=audit) as certify:
                outcome = policy.execute(env.task)
            self.assertTrue(outcome.success)
            self.assertEqual(env.arm_plan['approach_clearance_m'], .09)
            self.assertIn('GRASP_TARGET_INACCURATE', env.arm_plan['rejected_approaches'][1]['error'])
            self.assertEqual(certify.call_count, 1)
            self.assertEqual(env.box.squeeze, .003)
            self.assertEqual(env.task.base_obstacle_clearance_m, .08)
            self.assertEqual(env.task.arm_obstacle_clearance_m, .08)
            self.assertEqual(env.task.execution_obstacle_clearance_m, .03)


if __name__ == '__main__':
    unittest.main()
