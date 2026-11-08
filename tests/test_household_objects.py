from dataclasses import replace
import math
import unittest

import numpy as np

from grutopia_extension.interaction_clearance.task import ClearanceTask, SceneObstacle
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox
from grutopia_extension.manipulation.household_objects import grasp_height, object_parts


class HouseholdGeometryTests(unittest.TestCase):
    def test_tall_box_grasp_height_tracks_its_upper_side(self):
        self.assertEqual(grasp_height('box', (0.40, 0.50, 0.28)), 0.25)
        self.assertAlmostEqual(grasp_height('box', (0.48, 0.58, 0.34)), 0.31)

    def test_parts_stay_inside_declared_collision_envelope(self):
        dimensions = np.array((0.42, 0.48, 0.38))
        for kind in ('box', 'bin', 'stool'):
            parts = object_parts(kind, dimensions)
            points = np.concatenate([part.vertices for part in parts])
            np.testing.assert_allclose(points.min(axis=0), (-0.21, -0.24, 0), atol=1e-10)
            np.testing.assert_allclose(points.max(axis=0), (0.21, 0.24, 0.38), atol=1e-10)

    def test_bin_has_open_top_and_separate_bottom(self):
        parts = object_parts('bin', (0.40, 0.40, 0.38))
        walls = [part for part in parts if part.name.startswith('Wall')]
        floors = [part for part in parts if part.name.startswith('Floor')]
        self.assertEqual(len(walls), 20)
        self.assertEqual(len(floors), 20)
        self.assertTrue(all(min(np.linalg.norm(np.asarray(part.vertices)[:, :2], axis=1)) > 0.17 for part in walls))
        self.assertTrue(all(max(point[2] for point in part.vertices) == 0.02 for part in floors))

    def test_stool_has_four_legs_and_grips_the_seat(self):
        dimensions = (0.38, 0.46, 0.34)
        parts = object_parts('stool', dimensions)
        self.assertEqual([part.name for part in parts], ['Seat', 'Leg0', 'Leg1', 'Leg2', 'Leg3'])
        seat = parts[0]
        height = grasp_height('stool', dimensions)
        self.assertGreater(height, min(point[2] for point in seat.vertices))
        self.assertLess(height, max(point[2] for point in seat.vertices))
        BimanualBox(dimensions, 1.4, height, 0.003, 'stool').validate()


class ObstacleMarginTests(unittest.TestCase):
    def test_stronger_margins_and_obstacle_yaw_roundtrip(self):
        task = ClearanceTask(base_obstacle_clearance_m=0.08, arm_obstacle_clearance_m=0.08,
                             execution_obstacle_clearance_m=0.03, object_kind='bin')
        self.assertEqual(ClearanceTask.from_dict(task.to_dict()), task)
        obstacle = SceneObstacle('AngledCabinet', (3.0, 2.0, 0.5), (0.8, 0.3, 1.0), math.pi / 4)
        task = replace(task, scene_obstacles=task.scene_obstacles + (obstacle,))
        self.assertEqual(ClearanceTask.from_dict(task.to_dict()).scene_obstacles[-1], obstacle)

    def test_cannot_weaken_runtime_or_planning_minima(self):
        for values in ({'base_obstacle_clearance_m': 0.039}, {'arm_obstacle_clearance_m': 0.014},
                       {'execution_obstacle_clearance_m': 0.004}, {'execution_obstacle_clearance_m': 0.02}):
            with self.assertRaises(ValueError):
                replace(ClearanceTask(), **values).validate()

    def test_oriented_obstacle_cannot_intrude_into_passage(self):
        obstacle = SceneObstacle('Intrusion', (1.0, 0.80, 0.4), (1.2, 0.12, 0.8), math.pi / 2)
        with self.assertRaises(ValueError):
            replace(ClearanceTask(), scene_obstacles=(obstacle,)).validate()


if __name__ == '__main__':
    unittest.main()
