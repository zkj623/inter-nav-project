import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox, BimanualUrdfKinematics, opposed_side_contact


class BimanualBoxTests(unittest.TestCase):
    def test_failed_primary_ik_uses_one_bounded_torso_reseed(self):
        kinematics = object.__new__(BimanualUrdfKinematics)
        kinematics.lower = -np.ones(15)
        kinematics.upper = np.ones(15)

        def forward(positions):
            transform = np.eye(4)
            transform[0, 3] = positions[0]
            return transform, transform

        kinematics.forward = forward
        with patch('grutopia_extension.manipulation.bimanual_floor_lift.least_squares') as solver:
            solver.side_effect = [SimpleNamespace(x=np.r_[0.1, np.zeros(14)]), SimpleNamespace(x=np.zeros(15))]
            solution = kinematics.solve(np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), np.zeros(15), reseed_torso=True)
        self.assertEqual(solver.call_count, 2)
        np.testing.assert_allclose(solution, np.zeros(15))
        self.assertTrue(kinematics.last_solve_diagnostics[-1]['torso_reseeded'])
        self.assertEqual(solver.call_args_list[1].args[1][0], -0.2)

    def test_valid_primary_ik_does_not_change_to_an_alternate_solution(self):
        kinematics = object.__new__(BimanualUrdfKinematics)
        kinematics.lower = -np.ones(15)
        kinematics.upper = np.ones(15)
        kinematics.forward = lambda positions: (np.eye(4), np.eye(4))
        with patch('grutopia_extension.manipulation.bimanual_floor_lift.least_squares') as solver:
            solver.return_value.x = np.zeros(15)
            kinematics.solve(np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), np.zeros(15), reseed_torso=True)
        self.assertEqual(solver.call_count, 1)
        self.assertFalse(kinematics.last_solve_diagnostics[0]['torso_reseeded'])

    def test_planning_upper_limits_never_expand_physical_limits(self):
        kinematics = object.__new__(BimanualUrdfKinematics)
        kinematics.lower = -np.ones(15)
        kinematics.upper = np.ones(15)
        kinematics.forward = lambda positions: (np.eye(4), np.eye(4))
        with patch('grutopia_extension.manipulation.bimanual_floor_lift.least_squares') as solver:
            solver.return_value.x = np.zeros(15)
            kinematics.solve(np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), np.zeros(15), upper_limits=np.r_[0.0, np.full(14, 5.0)])
        actual_upper = solver.call_args.kwargs['bounds'][1]
        self.assertLess(actual_upper[0], 0)
        self.assertTrue(np.all(actual_upper[1:] < 1.0))

    def test_empty_planning_interval_is_rejected(self):
        kinematics = object.__new__(BimanualUrdfKinematics)
        kinematics.lower = -np.ones(15)
        kinematics.upper = np.ones(15)
        with self.assertRaises(ValueError):
            kinematics.solve(np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), np.zeros(15), upper_limits=np.full(15, -2.0))

    def test_box_is_large_and_grasp_point_is_on_the_side(self):
        box = BimanualBox()
        box.validate()
        self.assertGreaterEqual(box.dimensions[1], 0.5)
        self.assertLess(box.grasp_height, box.dimensions[2])

    def test_invalid_box_is_rejected(self):
        for box in (BimanualBox(mass=0), BimanualBox(grasp_height=0.5), BimanualBox(squeeze=0.2)):
            with self.assertRaises(ValueError):
                box.validate()

    def test_both_hands_must_push_from_opposite_sides(self):
        self.assertTrue(opposed_side_contact(((0, 20, -5), (0, -20, -5))))
        self.assertFalse(opposed_side_contact(((0, 20, -5), (0, 0, 0))))
        self.assertFalse(opposed_side_contact(((0, 20, -5), (0, 20, -5))))

    def test_bad_contact_measurements_are_rejected(self):
        self.assertFalse(opposed_side_contact(((0, float('nan'), 0), (0, -20, 0))))
        self.assertFalse(opposed_side_contact((0, 20, 0)))


if __name__ == '__main__':
    unittest.main()
