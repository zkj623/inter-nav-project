from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy, BaseMovePolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox


class ReleaseSelectionTests(unittest.TestCase):
    def test_placement_uses_only_reachable_and_fully_certified_release(self):
        environment = SimpleNamespace(task=ClearanceTask(), box=BimanualBox(), arm_approach_clearance_m=0.13,
            base=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0)))),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((0.65, 0.0, 0.22)), np.array((1.0, 0.0, 0.0, 0.0)))),
            robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(19)), control_indices=np.arange(19),
            rotation=lambda quaternion: np.eye(3), recorder=SimpleNamespace(event=Mock()))
        checker = SimpleNamespace(relative_box=np.eye(4), details=lambda pose: {'accepted': True, 'minimum_wall_clearance_m': 0.10})
        active = []

        def targets(mode, center, initial, clearance):
            active.append(clearance)
            if clearance == 0.13:
                raise RuntimeError('unreachable')
            return [np.zeros(19)], ['release'], [4.0]

        def certify(*arguments):
            if active[-1] == 0.11:
                raise RuntimeError('collision')
            return [{'minimum_wall_clearance_m': 0.10}]

        policy = BaseMovePolicy(environment)
        with patch('grutopia_extension.interaction_clearance.primitives.FrozenBaseCollisionChecker', return_value=checker), \
                patch('grutopia_extension.interaction_clearance.primitives.placement_candidates', return_value=[np.array((0.65, 0.70))]), \
                patch.object(ArmPlanPolicy, 'targets', side_effect=targets), \
                patch.object(ArmPlanPolicy, 'certify', side_effect=certify), \
                patch.object(policy, 'plan', return_value=[np.zeros(3), np.array((0.0, 0.7, 0.0))]) as path:
            policy.select_placement()
        self.assertEqual(active, [0.13, 0.11, 0.09])
        self.assertEqual(environment.arm_approach_clearance_m, 0.09)
        path.assert_called_once()
        event = environment.recorder.event.call_args.args[1]
        self.assertTrue(event['lowering_checked'])
        self.assertTrue(any(row['reason'] == 'collision' for row in event['rejected']))


if __name__ == '__main__':
    unittest.main()
