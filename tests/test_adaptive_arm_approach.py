from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox


class AdaptiveApproachTests(unittest.TestCase):
    def environment(self, output):
        return SimpleNamespace(task=ClearanceTask(), interaction_state='ready', arm_plan=None, box=BimanualBox(),
            base=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0)))),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((0.65, 0.0, 0.0001)), np.array((1.0, 0.0, 0.0, 0.0)))),
            robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(19)), control_indices=np.arange(19),
            rotation=lambda quaternion: np.eye(3), output=output, elapsed_steps=0, recorder=SimpleNamespace(event=Mock()))

    def test_reachable_but_uncertified_candidates_are_not_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = self.environment(Path(temporary))
            policy = ArmPlanPolicy(environment)
            targets = ([np.zeros(19)], ['approach'], [5.0])
            audit = [{'phase': 'approach', 'minimum_noncontact_clearance_m': 0.02, 'minimum_wall_clearance_m': 0.08}]
            with patch.object(policy, 'targets', return_value=targets) as solve, \
                    patch.object(policy, 'certify', side_effect=[RuntimeError('collision'), RuntimeError('collision'), audit]):
                outcome = policy.execute(environment.task)
            self.assertTrue(outcome.success)
            self.assertEqual([call.args[3] for call in solve.call_args_list], [0.13, 0.11, 0.09])
            self.assertEqual(environment.arm_approach_clearance_m, 0.09)
            self.assertEqual(len(environment.arm_plan['rejected_approaches']), 2)

    def test_all_unsafe_candidates_leave_no_executable_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = self.environment(Path(temporary))
            policy = ArmPlanPolicy(environment)
            with patch.object(policy, 'targets', return_value=([np.zeros(19)], ['approach'], [5.0])), \
                    patch.object(policy, 'certify', side_effect=RuntimeError('collision')):
                outcome = policy.execute(environment.task)
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.reason, 'ARM_PLAN_REJECTED')
            self.assertIsNone(environment.arm_plan)

    def test_release_fallback_is_certified_and_does_not_change_safety_margins(self):
        for collision in (False, True):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temporary:
                environment = self.environment(Path(temporary))
                environment.task = ClearanceTask(base_obstacle_clearance_m=0.08,
                    arm_obstacle_clearance_m=0.08, execution_obstacle_clearance_m=0.03)
                environment.interaction_state = 'holding'
                environment.selected_place_xy = np.array((0.65, 0.0))
                policy = ArmPlanPolicy(environment)
                attempted = []

                def solve(mode, center, initial, clearance):
                    attempted.append(clearance)
                    if clearance > 0.04:
                        raise RuntimeError('unreachable')
                    return [np.zeros(19)], ['release'], [4.0]

                audit = [{'minimum_noncontact_clearance_m': 0.033, 'minimum_wall_clearance_m': 0.198}]
                with patch.object(policy, 'targets', side_effect=solve), \
                        patch.object(policy, 'certify', side_effect=RuntimeError('collision') if collision else None,
                                     return_value=audit) as certify:
                    outcome = policy.execute(environment.task, mode='place')
                self.assertEqual(outcome.success, not collision)
                self.assertEqual(attempted[-1], 0.04)
                self.assertEqual(certify.call_count, 1)
                self.assertEqual(environment.task.base_obstacle_clearance_m, 0.08)
                self.assertEqual(environment.task.arm_obstacle_clearance_m, 0.08)
                self.assertEqual(environment.task.execution_obstacle_clearance_m, 0.03)
                if collision:
                    self.assertIsNone(environment.arm_plan)
                else:
                    self.assertEqual(environment.arm_plan['approach_clearance_m'], 0.04)


if __name__ == '__main__':
    unittest.main()
