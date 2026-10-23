from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from grutopia_extension.interaction_clearance.placement import placement_candidates, robot_passage_metrics
from grutopia_extension.interaction_clearance.task import ClearanceTask, Region, evaluate_clearance, yaw_rotation


class PlacementTests(unittest.TestCase):
    def test_candidates_preserve_allowed_region_and_clearance(self):
        task = ClearanceTask()
        candidates = placement_candidates(task, yaw_rotation(0))
        self.assertGreater(len(candidates), 1)
        np.testing.assert_allclose(candidates[0], task.place_xy)
        for position in candidates:
            metrics = evaluate_clearance(task, (*position, 0.0001), yaw_rotation(0), 0, 0, 0, True, 1)
            self.assertTrue(metrics['success'])

    def test_rotated_box_candidates_use_rotated_footprint(self):
        task = replace(ClearanceTask(), placement=Region((0.1, 0.5), (1.3, 1.8)), place_xy=(0.7, 1.2))
        rotation = yaw_rotation(0.7)
        candidates = placement_candidates(task, rotation)
        self.assertTrue(candidates)
        for position in candidates:
            self.assertTrue(evaluate_clearance(task, (*position, 0.0001), rotation, 0, 0, 0, True, 1)['success'])

    def test_too_small_region_has_no_candidate(self):
        task = replace(ClearanceTask(), placement=Region((0.6, 0.6), (0.7, 0.8)))
        self.assertEqual(placement_candidates(task, np.eye(3)), [])

    def test_robot_must_not_remain_in_passage(self):
        model = SimpleNamespace(vertices={'body': np.array([[-0.2, -0.2, 0], [0.2, 0.2, 1]])})
        body = np.eye(4)
        body[:2, 3] = (0.8, 0)
        self.assertFalse(robot_passage_metrics(ClearanceTask(), model, {'body': body})['robot_clear_of_passage'])
        body[1, 3] = 1.0
        self.assertTrue(robot_passage_metrics(ClearanceTask(), model, {'body': body})['robot_clear_of_passage'])

    def test_return_protocol_cannot_be_non_boolean(self):
        with self.assertRaises(ValueError):
            replace(ClearanceTask(), return_to_start=1).validate()


if __name__ == '__main__':
    unittest.main()
