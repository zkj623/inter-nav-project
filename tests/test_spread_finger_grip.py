from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import fcl
import numpy as np

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy, check_configuration, side_gripper_targets
from grutopia_extension.interaction_clearance.task import ClearanceTask
import test_lift_refinement as lift_fixtures


class SpreadGripTests(unittest.TestCase):
    def planning_environment(self, kind):
        task = ClearanceTask(object_kind=kind, dimensions=(.34, .42, .28), mass_kg=1.1)
        return SimpleNamespace(task=task, box=SimpleNamespace(dimensions=task.dimensions, squeeze=.003, grasp_height=.265),
            solve_both=Mock(return_value=np.zeros(15)), initial_control=np.zeros(19), rotation=lambda q: np.eye(3),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((.72, 0., .2201)), np.array((1., 0., 0., 0.)))))

    def test_stool_grasp_lift_and_place_keep_aperture_and_normal_preload(self):
        env = self.planning_environment('stool')
        planner = ArmPlanPolicy(env)
        for mode in ('grasp', 'place'):
            with patch.object(planner, 'check_squeeze_target'):
                targets, phases, _ = planner.targets(mode, np.array((.72, 0., .2201)), np.zeros(19))
            for target, phase in zip(targets, phases):
                np.testing.assert_array_equal(target[15:], np.zeros(4) if phase == 'retract' else [.03, -.03, .03, -.03])
        self.assertEqual(env.box.squeeze, .003)
        self.assertTrue(any(abs(c.args[1] - (.42 / 2 - .003)) < 1e-12 for c in env.solve_both.call_args_list))

    def test_box_and_bin_gripper_commands_remain_closed(self):
        for kind in ('box', 'bin'):
            env = self.planning_environment(kind)
            with patch.object(ArmPlanPolicy, 'check_squeeze_target'):
                targets, _, _ = ArmPlanPolicy(env).targets('grasp', np.array((.72, 0., .0001)), np.zeros(19))
            self.assertTrue(all(np.array_equal(t[15:], np.zeros(4)) for t in targets))

    def test_replanned_lift_does_not_silently_close_the_fingers(self):
        builder = lift_fixtures.LiftRefinementTests()
        with tempfile.TemporaryDirectory() as folder:
            env, policy = builder.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', return_value=builder.good_audit()):
                policy.lift_segment(.2201, 5)
            np.testing.assert_array_equal(policy.interpolate.call_args.args[0][15:], [.03, -.03, .03, -.03])
            self.assertEqual(policy.interpolate.call_args.args[1:], (.8, 'lift'))

    def test_opened_finger_near_obstacle_is_rejected_when_closed_would_pass(self):
        positions_seen = []
        def forward(positions, base):
            positions_seen.append(positions)
            return positions
        model = SimpleNamespace(fcl=fcl, forward=forward, box_clearances=lambda *a: {'body': 1.},
                                self_intersections=lambda t: [])
        model.objects = lambda t: {'finger': fcl.CollisionObject(fcl.Box(.01, .01, .01),
            fcl.Transform(np.eye(3), np.array((t['left_gripper_finger_joint1'], 0., 0.))))}
        env = SimpleNamespace(task=ClearanceTask(object_kind='stool', arm_obstacle_clearance_m=.08),
            collision_model=model, bimanual=SimpleNamespace(names=[f'joint{i}' for i in range(15)]),
            box=SimpleNamespace(dimensions=(.3, .3, .3)), wall_objects={'wall': fcl.CollisionObject(fcl.Box(.02, .2, .2),
            fcl.Transform(np.eye(3), np.array((.12, 0., 0.))))})
        box = np.eye(4); box[0, 3] = 2.
        self.assertGreater(check_configuration(env, np.zeros(19), np.eye(4), box)['minimum_wall_clearance_m'], .08)
        opened = np.r_[np.zeros(15), side_gripper_targets(env.task)]
        with self.assertRaisesRegex(RuntimeError, 'PLANNED_COLLISION'):
            check_configuration(env, opened, np.eye(4), box)
        self.assertEqual(positions_seen[-1]['right_gripper_finger_joint2'], -.03)

    def test_stool_cannot_be_certified_without_finger_configuration(self):
        env = SimpleNamespace(task=SimpleNamespace(object_kind='stool'), collision_model=SimpleNamespace(forward=Mock()),
                              bimanual=SimpleNamespace(names=[f'joint{i}' for i in range(15)]))
        with self.assertRaisesRegex(RuntimeError, 'STOOL_GRIPPER_TARGETS_MISSING'):
            check_configuration(env, np.zeros(15), np.eye(4), np.eye(4))
        env.collision_model.forward.assert_not_called()


if __name__ == '__main__':
    unittest.main()
