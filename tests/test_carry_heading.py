import math
from types import SimpleNamespace
import unittest

import numpy as np

from grutopia_extension.interaction_clearance.base_planner import SE2GridPlanner
from grutopia_extension.interaction_clearance.motion_geometry import angle_error
from grutopia_extension.interaction_clearance.placement import carrying_base_pose
from grutopia_extension.interaction_clearance.primitives import BaseMovePolicy
from grutopia_extension.interaction_clearance.task import yaw_rotation


class CarryHeadingTests(unittest.TestCase):
    def test_rotation_adjusts_base_goal_to_preserve_object_destination(self):
        goal = carrying_base_pose(np.zeros(3), np.array((0.65, 0, 0.22)), np.ones(2), math.pi / 2)
        np.testing.assert_allclose(goal, (1.0, 0.35, math.pi / 2))
        predicted_object = goal[:2] + yaw_rotation(goal[2])[:2, :2] @ np.array((0.65, 0))
        np.testing.assert_allclose(predicted_object, np.ones(2))

    def test_zero_rotation_matches_original_translation_target(self):
        start = np.array((0.2, -0.3, 0.4))
        position = np.array((0.8, 0.1, 0.2))
        destination = np.array((0.6, 2.0))
        expected = np.r_[start[:2] + destination - position[:2], start[2]]
        np.testing.assert_allclose(carrying_base_pose(start, position, destination), expected, atol=1e-12)

    def test_heading_astar_finds_required_mid_route_rotation(self):
        target_yaw = math.radians(15)

        def free(pose):
            if abs(pose[2]) > 1e-6 and pose[0] < 0.35:
                return False
            if abs(pose[2]) < 1e-6 and pose[0] > 0.75:
                return False
            return True

        planner = SE2GridPlanner(free, ((-0.2, -0.4), (1.2, 0.4)), yaw_offsets=(0, target_yaw), yaw_cost_m_per_rad=1.0)
        path = planner.plan(np.zeros(3), np.array((1.0, 0, target_yaw)))
        self.assertTrue(path.used_search)
        turns = [(first, second) for first, second in zip(path.waypoints, path.waypoints[1:])
                 if abs(angle_error(second[2], first[2])) > 1e-7]
        self.assertTrue(turns)
        for first, second in turns:
            self.assertGreaterEqual(first[0], 0.35)
            self.assertLessEqual(first[0], 0.75)
            np.testing.assert_allclose(first[:2], second[:2])

    def test_invalid_heading_graph_or_motion_cost_is_rejected(self):
        for offsets in ((1, 0.2), (0, 0), (0, float('nan'))):
            with self.assertRaises(ValueError):
                SE2GridPlanner(lambda pose: True, ((-1, -1), (1, 1)), yaw_offsets=offsets)
        with self.assertRaises(ValueError):
            SE2GridPlanner(lambda pose: True, ((-1, -1), (1, 1)), yaw_cost_m_per_rad=0)

    def fake_environment(self, initial_yaw=0.0):
        state = {'yaw': initial_yaw, 'commands': []}

        def quaternion():
            return np.array((math.cos(state['yaw'] / 2), 0, 0, math.sin(state['yaw'] / 2)))

        environment = SimpleNamespace(command=np.zeros(19), control_indices=np.arange(19), base_twist=np.zeros(3),
            control_dt=0.02, base=SimpleNamespace(get_world_pose=lambda: (np.zeros(3), quaternion()),
                                                get_linear_velocity=lambda: np.zeros(3)),
            object=SimpleNamespace(get_world_pose=lambda: (np.array((0.65, 0, 0.22)), quaternion())),
            object_bottom=lambda: 0.22, finger_force_vectors=np.zeros((4, 3)))

        def advance(joints, phase, progress):
            state['commands'].append(environment.base_twist[2])
            state['yaw'] += environment.base_twist[2] * environment.control_dt
            local_forces = np.array(((0, 1, 0), (0, 1, 0), (0, -1, 0), (0, -1, 0)))
            environment.finger_force_vectors = local_forces @ yaw_rotation(state['yaw']).T

        environment.advance = advance
        return environment, state

    def test_carrying_pure_rotation_tracks_slowly_without_false_yaw_rejection(self):
        environment, state = self.fake_environment()
        target = np.array((0, 0, math.radians(15)))
        BaseMovePolicy(environment).drive_segment(target, carrying=True, rotation_segment=True)
        self.assertLess(abs(angle_error(target[2], state['yaw'])), 0.008)
        self.assertLessEqual(max(abs(value) for value in state['commands']), 0.0800001)
        np.testing.assert_allclose(environment.base_twist, np.zeros(3))

    def test_rotation_tolerates_small_residual_pivot_position_error(self):
        environment, state = self.fake_environment()
        original_pose = environment.base.get_world_pose
        environment.base.get_world_pose = lambda: (np.array((0.004, 0, 0)), original_pose()[1])
        target = np.array((0, 0, math.radians(15)))
        BaseMovePolicy(environment).drive_segment(target, carrying=True, rotation_segment=True)
        self.assertLess(abs(angle_error(target[2], state['yaw'])), 0.008)
        self.assertLessEqual(max(abs(value) for value in state['commands']), 0.0800001)

    def test_translation_keeps_original_yaw_correction_with_small_heading_error(self):
        environment, state = self.fake_environment(initial_yaw=0.02)

        def stop_after_command(joints, phase, progress):
            state['commands'].append(environment.base_twist[2])
            raise RuntimeError('STOP_AFTER_FIRST_COMMAND')

        environment.advance = stop_after_command
        with self.assertRaisesRegex(RuntimeError, 'STOP_AFTER_FIRST_COMMAND'):
            BaseMovePolicy(environment).drive_segment(np.array((0.1, 0, 0)), carrying=True)
        self.assertAlmostEqual(state['commands'][0], -0.3 * environment.control_dt)

    def test_translation_still_rejects_unexpected_carrying_yaw(self):
        environment, state = self.fake_environment(initial_yaw=0.08)
        with self.assertRaisesRegex(RuntimeError, 'CARRYING_YAW_TRACKING_DEVIATION'):
            BaseMovePolicy(environment).drive_segment(np.array((0.1, 0, 0)), carrying=True)
        self.assertEqual(state['commands'], [])


if __name__ == '__main__':
    unittest.main()
