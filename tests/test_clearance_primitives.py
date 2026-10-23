from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from grutopia_extension.interaction_clearance.motion_geometry import (
    angle_error, base_waypoints, bounded_swerve_targets, grasp_base_pose, grasp_frames, opposed_object_contact, sample_base_path,
)
from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext, load_skill
from grutopia_extension.interaction_clearance.task import ClearanceTask, yaw_rotation
from grutopia_extension.interaction_clearance.benchmark import pose_suite, suite_cases
from grutopia.demo.evolve_r1lite_clearance_skill import generation_prompt
from grutopia_extension.interaction_clearance.primitives import ArmExecutePolicy, BaseMovePolicy


class PoseGeometryTests(unittest.TestCase):
    def test_task_round_trip_preserves_four_initial_pose_fields(self):
        task = replace(ClearanceTask(), robot_xy=(-0.15, 0.05), robot_yaw_rad=-0.3, object_xy=(0.72, 0.08), object_yaw_rad=0.2)
        task.validate()
        self.assertEqual(ClearanceTask.from_dict(task.to_dict()), task)

    def test_grasp_pose_is_object_relative_not_fixed_world_axes(self):
        yaw = math.radians(35)
        center = np.array((0.8, 0.1, 0.0))
        pose = grasp_base_pose(center, yaw, 0.65)
        relative = yaw_rotation(yaw).T @ (center - np.r_[pose[:2], 0])
        np.testing.assert_allclose(relative, (0.65, 0, 0), atol=1e-12)
        self.assertAlmostEqual(pose[2], yaw)

    def test_both_tool_poses_rotate_with_box(self):
        yaw = 0.4
        center = np.array((0.7, -0.1, 0.0))
        left, right, left_rotation, right_rotation = grasp_frames(center, yaw, 0.25, 0.25)
        np.testing.assert_allclose(yaw_rotation(yaw).T @ (left - center), (0, 0.25, 0.25), atol=1e-12)
        np.testing.assert_allclose(yaw_rotation(yaw).T @ (right - center), (0, -0.25, 0.25), atol=1e-12)
        for rotation in (left_rotation, right_rotation):
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_grip_force_is_checked_in_object_frame(self):
        yaw = math.radians(100)
        forces = np.array(((0, 8, 0), (0, -8, 0))) @ yaw_rotation(yaw).T
        self.assertTrue(opposed_object_contact(forces, yaw))
        self.assertFalse(opposed_object_contact(forces, 0))

    def test_angles_wrap_and_base_sampling_has_no_large_jump(self):
        start = np.array((-0.1, 0.1, math.radians(179)))
        target = np.array((0.0, -0.1, math.radians(-179)))
        points = base_waypoints(start, target, 'retreat')
        samples = sample_base_path(points)
        self.assertAlmostEqual(angle_error(target[2], start[2]), math.radians(2))
        for first, second in zip(samples, samples[1:]):
            self.assertLessEqual(np.linalg.norm(second[:2] - first[:2]), 0.01000001)
            self.assertLessEqual(abs(angle_error(second[2], first[2])), math.radians(1.00001))

    def test_rotated_box_target_footprint_must_fit(self):
        task = replace(ClearanceTask(), object_yaw_rad=math.pi / 4)
        with self.assertRaises(ValueError):
            task.validate()

    def test_suite_varies_robot_and_box_poses_by_more_than_millimeters(self):
        suite = pose_suite()
        tasks = [layout['task'] for layout in suite['layouts']]
        self.assertGreater(np.ptp([task['robot_xy'][0] for task in tasks]), 0.10)
        self.assertGreater(np.ptp([task['object_xy'][0] for task in tasks]), 0.05)
        self.assertGreater(np.ptp([task['robot_yaw_rad'] for task in tasks]), math.radians(40))
        self.assertGreater(np.ptp([task['object_yaw_rad'] for task in tasks]), math.radians(30))
        self.assertEqual([len(suite_cases(suite, split)) for split in ('train', 'validation', 'heldout')], [3, 2, 2])

    def test_primitive_generation_prompt_exposes_no_heldout_poses(self):
        suite = pose_suite()
        prompt = generation_prompt(suite, [], 1)
        self.assertIn("ctx.execute('base_move'", prompt)
        self.assertIn("ctx.execute('arm_plan'", prompt)
        for layout in suite['layouts']:
            if layout['split'] != 'train':
                self.assertNotIn(layout['layout_id'], prompt)


class SteeringRangeTests(unittest.TestCase):
    positions = ((0.29183, -0.0014897), (-0.16817, 0.23851), (-0.16817, -0.24149))

    def test_winding_boundary_uses_equivalent_reverse_wheel_solution(self):
        yaw = 0.12
        command = (0.08 * math.cos(yaw), 0.08 * math.sin(yaw), 0.0)
        current = np.full(3, 2 * math.pi - 0.02)
        targets, velocities = bounded_swerve_targets(command, current, self.positions)
        np.testing.assert_allclose(targets, yaw + math.pi)
        np.testing.assert_allclose(velocities, 0.0)
        self.assertLess(float(np.max(np.abs(targets))), 2 * math.pi)

    def test_aligned_backwards_modules_preserve_forward_translation(self):
        targets, velocities = bounded_swerve_targets((0.07, 0, 0), np.full(3, math.pi), self.positions)
        np.testing.assert_allclose(targets, math.pi)
        np.testing.assert_allclose(velocities, -1.0)

    def test_stopped_modules_hold_measured_angles(self):
        current = np.array((-5.5, 0.4, 5.5))
        targets, velocities = bounded_swerve_targets((0, 0, 0), current, self.positions)
        np.testing.assert_array_equal(targets, current)
        np.testing.assert_array_equal(velocities, np.zeros(3))

    def test_aligned_equivalents_reconstruct_each_module_velocity(self):
        command = np.array((0.03, -0.02, 0.04))
        current = np.array((-6.1, 0.8, 6.1))
        targets, _ = bounded_swerve_targets(command, current, self.positions)
        aligned, velocities = bounded_swerve_targets(command, targets, self.positions)
        measured = 0.07 * velocities[:, None] * np.column_stack((np.cos(aligned), np.sin(aligned)))
        positions = np.asarray(self.positions)
        expected = np.column_stack((command[0] - command[2] * positions[:, 1],
                                     command[1] + command[2] * positions[:, 0]))
        np.testing.assert_allclose(measured, expected, atol=1e-12)

    def test_sweep_never_commands_out_of_range_or_over_speed(self):
        current = np.array((-6.2, 0.0, 6.2))
        for yaw in np.linspace(-math.pi, math.pi, 101):
            targets, velocities = bounded_swerve_targets((math.cos(yaw), math.sin(yaw), 0.5), current, self.positions)
            self.assertLess(float(np.max(np.abs(targets))), 2 * math.pi)
            self.assertLessEqual(float(np.max(np.abs(velocities))), 2.00000001)
            current = targets

    def test_invalid_state_or_parameters_fail_before_drive(self):
        for current in ((7, 0, 0), (math.nan, 0, 0), (0, 0)):
            with self.assertRaises(ValueError):
                bounded_swerve_targets((0.1, 0, 0), current, self.positions)
        for radius in (0, math.nan):
            with self.assertRaises(ValueError):
                bounded_swerve_targets((0.1, 0, 0), (0, 0, 0), self.positions, wheel_radius=radius)


class PrimitiveSkillTests(unittest.TestCase):
    def context(self, responses):
        calls = []

        class FakePolicy:
            def __init__(self, name):
                self.name = name

            def execute(self, task, **options):
                calls.append((self.name, options))
                return responses.get(self.name, Outcome(True, 'DONE'))

        names = ('base_move', 'arm_plan', 'grasp_lift', 'place_release')
        context = SkillContext(ClearanceTask(), {name: FakePolicy(name) for name in names},
                               lambda: Outcome(True, 'ROAD_CLEARED'), lambda: {}, lambda *args: None)
        return context, calls

    def test_baseline_uses_separate_base_plan_and_arm_execution(self):
        context, calls = self.context({})
        skill, _ = load_skill(Path(__file__).parents[1] / 'grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py')
        self.assertTrue(skill(context).success)
        self.assertEqual([name for name, _ in calls], ['base_move', 'arm_plan', 'grasp_lift', 'base_move', 'arm_plan', 'place_release'])

    def test_return_to_start_remains_an_explicit_optional_protocol(self):
        from dataclasses import replace

        context, calls = self.context({})
        context.task = replace(context.task, return_to_start=True)
        skill, _ = load_skill(Path(__file__).parents[1] / 'grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py')
        self.assertTrue(skill(context).success)
        self.assertEqual(len(calls), 7)
        self.assertEqual(calls[-1], ('base_move', {'target': 'home', 'route': 'auto'}))

    def test_arbitrary_base_and_arm_goals_cannot_be_injected(self):
        context, _ = self.context({})
        self.assertEqual(context.execute('base_move', target='unvalidated_world_pose').reason, 'INVALID_POLICY_ARGUMENTS')
        self.assertEqual(context.execute('arm_plan', mode='place', place_xy=(100, 100)).reason, 'INVALID_POLICY_ARGUMENTS')
        self.assertEqual(context.execute('grasp_lift', object_id='other').reason, 'INVALID_POLICY_ARGUMENTS')

    def test_non_retry_safe_failures_block_subsequent_motion(self):
        context, calls = self.context({'grasp_lift': Outcome(False, 'GRASP_LOST')})
        self.assertEqual(context.execute('grasp_lift').reason, 'GRASP_LOST')
        self.assertEqual(context.execute('base_move', target='placement').reason, 'UNSAFE_RETRY_REJECTED')
        self.assertEqual(len(calls), 1)

    def test_pre_motion_planning_rejection_allows_alternate_route(self):
        context, calls = self.context({'base_move': Outcome(False, 'BASE_PLAN_REJECTED', {'retry_safe': True})})
        self.assertEqual(context.execute('base_move', target='grasp').reason, 'BASE_PLAN_REJECTED')
        self.assertEqual(context.execute('base_move', target='grasp', route='retreat').reason, 'BASE_PLAN_REJECTED')
        self.assertEqual(len(calls), 2)


class PrimitiveSafetyTests(unittest.TestCase):
    def test_long_certified_segments_have_time_to_finish_without_speed_increase(self):
        self.assertGreater(BaseMovePolicy.segment_budget(np.zeros(3), np.array((3.0, 0, 0))), 37.5)
        self.assertEqual(BaseMovePolicy.segment_budget(np.zeros(3), np.array((0.1, 0, 0))), 30.0)

    def environment(self):
        return SimpleNamespace(
            task=ClearanceTask(), interaction_state='ready', arm_plan=None, base_twist=np.zeros(3), safety_ok=True,
            base=SimpleNamespace(get_world_pose=lambda: (np.array((0.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((0.65, 0.0, 0.0)), np.array((1.0, 0.0, 0.0, 0.0)))),
            robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(19)), control_indices=np.arange(19),
            start_base_position=np.zeros(3), start_base_quaternion=np.array((1.0, 0.0, 0.0, 0.0)))

    def test_base_target_respects_task_and_interaction_state(self):
        environment = self.environment()
        policy = BaseMovePolicy(environment)
        np.testing.assert_allclose(policy.target('grasp'), (0, 0, 0), atol=1e-12)
        with self.assertRaises(ValueError):
            policy.target('placement')
        environment.interaction_state = 'holding'
        np.testing.assert_allclose(policy.target('placement'), (0, 0.70, 0), atol=1e-12)
        with self.assertRaises(ValueError):
            policy.target('home')

    def test_collision_rejected_base_plan_never_actuates(self):
        policy = BaseMovePolicy(self.environment())
        with patch.object(policy, 'plan', side_effect=RuntimeError('PLANNED_COLLISION')), patch.object(policy, 'drive_segment') as drive:
            outcome = policy.execute(ClearanceTask(), target='grasp')
        self.assertEqual(outcome.reason, 'BASE_PLAN_REJECTED')
        self.assertTrue(outcome.details['retry_safe'])
        drive.assert_not_called()

    def test_executing_base_failure_is_not_retry_safe(self):
        policy = BaseMovePolicy(self.environment())
        with patch.object(policy, 'plan', return_value=[np.zeros(3), np.array((-0.1, 0, 0))]), patch.object(policy, 'drive_segment', side_effect=RuntimeError('timeout')):
            outcome = policy.execute(ClearanceTask(), target='grasp')
        self.assertEqual(outcome.reason, 'BASE_EXECUTION_FAILURE')
        self.assertFalse(outcome.details.get('retry_safe', False))

    def test_stale_arm_plan_is_discarded_before_execution(self):
        environment = self.environment()
        environment.arm_plan = {'mode': 'grasp', 'base': np.array((0.10, 0, 0)), 'object': np.array((0.65, 0, 0)),
                                'object_yaw': 0.0, 'initial': np.zeros(19)}
        policy = ArmExecutePolicy(environment, 'grasp')
        with patch.object(policy, 'interpolate') as execute:
            outcome = policy.execute(ClearanceTask())
        self.assertEqual(outcome.reason, 'STALE_ARM_PLAN')
        self.assertIsNone(environment.arm_plan)
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
