"""Equivalent support geometry must not change its witness with world coordinates."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import test_local_support_correction as fixtures
from grutopia_extension.interaction_clearance.primitives import SeatEdgeArmExecutePolicy


class ObjectFrameSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LocalSupportCorrectionTests.setUpClass()
        cls.states = json.loads((Path(__file__).parent / 'fixtures/r32_support_states.json').read_text())['states']

    def transforms(self):
        tiny = np.eye(4); tiny[0, 3] = 1e-10
        rigid = np.eye(4)
        rigid[:3, :3] = Rotation.from_euler('xyz', (.05, -.08, .3)).as_matrix()
        rigid[:3, 3] = (.1, -.2, .3)
        return (np.eye(4), tiny, rigid)

    def measurements(self, state, object_frame):
        env, poses, measured, box, _ = fixtures.LocalSupportCorrectionTests().environment(data=state)
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        return [policy.support_alignment({name: world @ pose for name, pose in poses.items()},
                    measured, world @ box, object_frame=object_frame) for world in self.transforms()]

    def test_all_eight_recorded_states_are_rigid_transform_invariant(self):
        self.assertEqual(len(self.states), 8)
        for index, state in enumerate(self.states):
            with self.subTest(index=index):
                rows = self.measurements(state, True)
                self.assertLess(np.ptp([row['support_x_error_m'] for row in rows]), 1e-8)
                self.assertTrue(all(row['support_query_frame'] == 'object' for row in rows))

    def test_legacy_world_query_exposes_the_recorded_coordinate_sensitivity(self):
        spreads = [np.ptp([row['support_x_error_m'] for row in self.measurements(state, False)])
                   for state in self.states]
        self.assertGreater(max(spreads), .0002)

    def test_contact_geometry_and_original_support_gates_remain_valid(self):
        for index, state in enumerate(self.states):
            with self.subTest(index=index):
                current, legacy = self.measurements(state, True), self.measurements(state, False)
                for row, old in zip(current, legacy):
                    points = np.asarray(row['lower_contact_points_object'])
                    gaps = np.asarray(row['support_query_gaps_m'])
                    self.assertTrue(np.all((gaps > 0.) & (gaps < .001)))
                    self.assertLess(np.max(np.abs(gaps - old['support_query_gaps_m'])), 3e-6)
                    self.assertGreater(points[0, 1], 0.)
                    self.assertLess(points[1, 1], 0.)
                    self.assertGreater(points[0, 1] - points[1, 1], .1)
                    self.assertTrue(np.all(np.abs(points[:, 0]) <= state['task']['dimensions'][0] / 2 + 1e-8))
                    np.testing.assert_allclose(points[:, 2], state['task']['dimensions'][2] - .06, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
