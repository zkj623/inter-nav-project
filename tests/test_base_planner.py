import math
import unittest

import numpy as np

from grutopia_extension.interaction_clearance.base_planner import PlanningError, SE2GridPlanner, pose_samples
from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext
from grutopia_extension.interaction_clearance.task import ClearanceTask


class BasePlannerTests(unittest.TestCase):
    def assert_safe(self, result, free):
        for first, second in zip(result.waypoints, result.waypoints[1:]):
            self.assertTrue(all(free(pose) for pose in pose_samples(first, second)))

    def test_open_path_and_long_segments_are_bounded(self):
        planner = SE2GridPlanner(lambda pose: True, ((-1, -1), (5, 1)), fixed_yaw=True)
        result = planner.plan((0, 0, 0.3), (4, 0, 0.3))
        self.assertFalse(result.used_search)
        np.testing.assert_allclose(result.waypoints[-1], (4, 0, 0.3))
        self.assertTrue(all(np.linalg.norm(second[:2] - first[:2]) <= 1.200001
                            for first, second in zip(result.waypoints, result.waypoints[1:])))

    def test_multiple_obstacles_trigger_search_not_fixed_axis_routes(self):
        def free(pose):
            return not (np.all(np.abs(pose[:2]) <= 0.3)
                        or np.all(np.abs(pose[:2] - (0.5, -0.4)) <= (0.25, 0.2)))

        result = SE2GridPlanner(free, ((-1.5, -1.5), (1.5, 1.5)), fixed_yaw=True).plan((-1, -1, 0.2), (1, 1, 0.2))
        self.assertTrue(result.used_search)
        self.assertGreater(result.expanded_nodes, 0)
        self.assert_safe(result, free)
        self.assertTrue(all(abs(pose[2] - 0.2) < 1e-7 for pose in result.waypoints))

    def test_empty_robot_changes_orientation_away_from_start(self):
        def free(pose):
            if pose[1] < 0.2 and abs(pose[2]) > 0.2:
                return False
            if 0.39 < pose[1] < 0.61 and abs(pose[2] - math.pi / 2) > 0.2:
                return False
            return True

        result = SE2GridPlanner(free, ((-0.2, -0.1), (0.2, 1.1))).plan((0, 0, 0), (0, 1, math.pi / 2))
        self.assertTrue(result.used_search)
        self.assert_safe(result, free)
        self.assertTrue(any(0.19 <= pose[1] <= 0.39 and abs(pose[2]) > 0.3 for pose in result.waypoints))

    def test_yaw_wrap_uses_short_turn(self):
        result = SE2GridPlanner(lambda pose: True, ((-1, -1), (1, 1))).plan(
            (0, 0, math.radians(179)), (0, 0, math.radians(-179)))
        samples = pose_samples(result.waypoints[0], result.waypoints[-1])
        self.assertLess(abs(samples[-1][2] - samples[0][2]), math.radians(3))

    def test_carried_object_cannot_request_yaw_rotation(self):
        with self.assertRaisesRegex(PlanningError, 'CARRYING_YAW_ROTATION_UNSUPPORTED'):
            SE2GridPlanner(lambda pose: True, ((-1, -1), (1, 1)), fixed_yaw=True).plan((0, 0, 0), (0.5, 0, 0.2))

    def test_fine_certification_rejects_between_grid_samples(self):
        def free(pose):
            return not (0.047 <= pose[0] <= 0.055 and abs(pose[1]) < 0.06)

        result = SE2GridPlanner(free, ((-0.3, -0.3), (0.4, 0.3)), fixed_yaw=True).plan((0, 0, 0), (0.2, 0, 0))
        self.assertTrue(result.used_search)
        self.assert_safe(result, free)

    def test_no_path_is_a_bounded_failure(self):
        with self.assertRaisesRegex(PlanningError, 'NO_PATH_IN_SEARCH_BOUNDS'):
            SE2GridPlanner(lambda pose: abs(pose[0]) > 0.1, ((-0.8, -0.4), (0.8, 0.4)),
                           fixed_yaw=True).plan((-0.5, 0, 0), (0.5, 0, 0))

    def test_expansion_limit_is_reported(self):
        with self.assertRaisesRegex(PlanningError, 'PLANNING_EXPANSION_LIMIT'):
            SE2GridPlanner(lambda pose: not np.all(np.abs(pose[:2]) < 0.2), ((-1, -1), (1, 1)),
                           fixed_yaw=True, max_expansions=1).plan((-0.5, -0.5, 0), (0.5, 0.5, 0))

    def test_whole_footprint_cannot_use_point_sized_gap(self):
        def free(pose, radius):
            upper = (-0.05 - radius <= pose[0] <= 0.05 + radius and pose[1] >= 0.15 - radius)
            lower = (-0.05 - radius <= pose[0] <= 0.05 + radius and pose[1] <= -0.15 + radius)
            return not (upper or lower)

        point = SE2GridPlanner(lambda pose: free(pose, 0), ((-1, -1), (1, 1)), fixed_yaw=True)
        self.assertFalse(point.plan((-0.5, 0, 0), (0.5, 0, 0)).used_search)
        with self.assertRaisesRegex(PlanningError, 'NO_PATH_IN_SEARCH_BOUNDS'):
            SE2GridPlanner(lambda pose: free(pose, 0.2), ((-1, -1), (1, 1)),
                           fixed_yaw=True).plan((-0.5, 0, 0), (0.5, 0, 0))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            SE2GridPlanner(lambda pose: True, ((1, 0), (0, 1)))
        with self.assertRaises(ValueError):
            SE2GridPlanner(lambda pose: True, ((-1, -1), (1, 1))).plan((0, 0, math.nan), (0, 0, 0))

    def test_skill_defaults_to_auto_planner(self):
        calls = []

        class Policy:
            def execute(self, task, **options):
                calls.append(options)
                return Outcome(True, 'DONE')

        context = SkillContext(ClearanceTask(), {'base_move': Policy()}, lambda: None, lambda: {}, lambda *args: None)
        self.assertTrue(context.execute('base_move', target='grasp').success)
        self.assertEqual(calls, [{'target': 'grasp', 'route': 'auto'}])


if __name__ == '__main__':
    unittest.main()
