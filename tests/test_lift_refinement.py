"""Synthetic control-flow checks; no physics or robot success is simulated here."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from grutopia_extension.interaction_clearance.primitives import ArmExecutePolicy, ArmPlanPolicy, LiftGripPathError
from grutopia_extension.interaction_clearance.task import ClearanceTask


class LiftRefinementTests(unittest.TestCase):
    def environment(self, directory, undershoot=0.0, lose_contact=False):
        position = np.array((.76, 0., .18385))
        task = ClearanceTask(object_kind='stool', dimensions=(.44, .52, .36), mass_kg=1.8)
        env = SimpleNamespace(task=task, box=SimpleNamespace(dimensions=task.dimensions, squeeze=.003, grasp_height=.345),
            command=np.zeros(19), control_indices=np.arange(19), output=Path(directory), elapsed_steps=0,
            object=SimpleNamespace(get_world_pose=lambda: (position.copy(), np.array((1., 0., 0., 0.)))),
            base=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array((1., 0., 0., 0.)))),
            rotation=lambda q: np.eye(3), recorder=SimpleNamespace(event=Mock()),
            finger_force_vectors=np.array(((0., 20., 0.), (0., 20., 0.), (0., -20., 0.), (0., -20., 0.))))

        def solve(center, pressed, height, seed, **kwargs):
            env.pending_height = float(center[2] + height - env.box.grasp_height)
            return np.zeros(15)

        def motion(target, duration, phase):
            position[2] = env.pending_height - undershoot
            env.elapsed_steps += 40
            if lose_contact:
                env.finger_force_vectors[:] = 0

        env.solve_both = Mock(side_effect=solve)
        policy = ArmExecutePolicy(env, 'grasp')
        policy.interpolate = Mock(side_effect=motion)
        return env, policy

    def violation(self, progress=.5):
        return LiftGripPathError({'grip_span_error_m': .001108, 'grip_span_tolerance_m': .001}, progress)

    def good_audit(self):
        return [{'minimum_noncontact_clearance_m': .041, 'minimum_wall_clearance_m': .296}]

    def test_already_feasible_segment_keeps_one_motion_and_original_target(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', return_value=self.good_audit()) as certify:
                policy.lift_segment(.2201, 5)
            self.assertEqual(certify.call_count, 1)
            self.assertEqual(policy.interpolate.call_count, 1)
            self.assertEqual(policy.interpolate.call_args.args[1:], (.8, 'lift'))
            self.assertAlmostEqual(env.object.get_world_pose()[0][2], .2201)

    def test_bad_full_segment_refines_before_motion_and_uses_new_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder, undershoot=.001)

            def certify(*args, lift_distance, **kwargs):
                if lift_distance > .025:
                    self.assertEqual(policy.interpolate.call_count, 0)
                    raise self.violation()
                return self.good_audit()

            with patch.object(ArmPlanPolicy, 'certify', side_effect=certify):
                policy.lift_segment(.2201, 5)
            self.assertEqual(policy.interpolate.call_count, 2)
            calls = env.solve_both.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertAlmostEqual(calls[1].args[2] - .345, .03625 / 2)
            self.assertAlmostEqual(calls[2].args[2] - .345, .03625 / 2 + .001)
            self.assertTrue(all(call.args[1] == .8 for call in policy.interpolate.call_args_list))
            last_replan = [c.args[1] for c in env.recorder.event.call_args_list if c.args[0] == 'lift_replan'][-1]
            self.assertAlmostEqual(last_replan['target_origin_height_m'], .2201)
            self.assertEqual(env.box.squeeze, .003)

    def test_initial_span_violation_never_refines(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', side_effect=self.violation(0.0)) as certify:
                with self.assertRaises(LiftGripPathError):
                    policy.lift_segment(.2201, 5)
            self.assertEqual(certify.call_count, 1)
            policy.interpolate.assert_not_called()

    def test_persistent_violation_is_bounded_without_any_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', side_effect=self.violation()) as certify:
                with self.assertRaises(LiftGripPathError):
                    policy.lift_segment(.2201, 5)
            self.assertEqual(certify.call_count, 4)
            policy.interpolate.assert_not_called()
            np.testing.assert_array_equal(env.command, np.zeros(19))

    def test_tiny_rejected_displacement_is_not_split_indefinitely(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', side_effect=self.violation()) as certify:
                with self.assertRaises(LiftGripPathError):
                    policy.lift_segment(.18785, 5)
            self.assertEqual(certify.call_count, 1)
            policy.interpolate.assert_not_called()

    def test_contact_loss_after_first_half_stops_without_second_half(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder, lose_contact=True)

            def certify(*args, lift_distance, **kwargs):
                if lift_distance > .025:
                    raise self.violation()
                return self.good_audit()

            with patch.object(ArmPlanPolicy, 'certify', side_effect=certify):
                with self.assertRaisesRegex(RuntimeError, 'GRASP_LOST_DURING_LIFT'):
                    policy.lift_segment(.2201, 5)
            self.assertEqual(policy.interpolate.call_count, 1)

    def test_other_collision_rejection_never_enters_grip_refinement(self):
        with tempfile.TemporaryDirectory() as folder:
            env, policy = self.environment(folder)
            with patch.object(ArmPlanPolicy, 'certify', side_effect=RuntimeError('PLANNED_COLLISION')) as certify:
                with self.assertRaisesRegex(RuntimeError, 'PLANNED_COLLISION'):
                    policy.lift_segment(.2201, 5)
            self.assertEqual(certify.call_count, 1)
            policy.interpolate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
