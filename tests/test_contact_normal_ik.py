"""Numerical solver tests, not physical robot trials."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.bimanual_floor_lift import BimanualUrdfKinematics


class ContactNormalIKTests(unittest.TestCase):
    def model(self):
        model = BimanualUrdfKinematics.__new__(BimanualUrdfKinematics)
        model.lower = -np.ones(15)
        model.upper = np.ones(15)

        def forward(q):
            tools = []
            for value in q[:2]:
                tool = np.eye(4)
                tool[:3, :3] = Rotation.from_euler('z', 10 * value).as_matrix()
                tool[1, 3] = value
                tools.append(tool)
            return tools

        model.forward = forward
        return model

    def solve(self, **kwargs):
        model = self.model()
        return model.solve((0, .01, 0), (0, -.01, 0), np.eye(3), np.eye(3), np.zeros(15), **kwargs)

    def test_normal_priority_preserves_pinching_precision_under_orientation_conflict(self):
        legacy = self.solve()
        precise = self.solve(contact_normal=(0, 1, 0))
        desired = np.array((.01, -.01))
        self.assertGreater(np.max(np.abs(legacy[:2] - desired)), .004)
        self.assertLess(np.max(np.abs(precise[:2] - desired)), .0002)
        self.assertTrue(np.all(np.abs(precise) < 1))

    def test_axis_scale_and_sign_do_not_change_the_normal_constraint(self):
        reference = self.solve(contact_normal=(0, 1, 0))
        for axis in ((0, 7, 0), (0, -1, 0)):
            np.testing.assert_allclose(self.solve(contact_normal=axis), reference, atol=1e-9)

    def test_invalid_normal_is_rejected_before_optimization(self):
        for axis in ((0, 0, 0), (0, float('nan'), 0), (1, 0), (0, float('inf'), 0)):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'Contact normal'):
                self.solve(contact_normal=axis)


if __name__ == '__main__':
    unittest.main()
