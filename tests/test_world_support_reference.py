"""World target persistence with the base drift measured in the failed R31 trial."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

import test_local_support_correction as fixtures
import test_individual_finger_calibration as native_fixtures
from grutopia_extension.interaction_clearance.primitives import SeatEdgeArmExecutePolicy


class WorldSupportReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LocalSupportCorrectionTests.setUpClass()
        cls.model = fixtures.LocalSupportCorrectionTests.model
        cls.kin = fixtures.LocalSupportCorrectionTests.kin
        cls.data = json.loads((Path(__file__).parent / 'fixtures/r31_base_drift.json').read_text())

    def environment(self):
        return fixtures.LocalSupportCorrectionTests().environment(data=self.data['initial'])

    @staticmethod
    def transform(pose):
        return native_fixtures.IndividualFingerCalibrationTests.transform(pose[:3], pose[3:])

    def test_saved_world_pose_survives_base_translation(self):
        env, _, _, box, _ = self.environment()
        first, second = self.data['base_pose_sequence'][:2]
        reference = tuple(self.transform(first) @ frame for frame in self.kin.forward(env.command[:15]))
        env.base.get_world_pose = lambda: (np.asarray(second[:3]), np.asarray(second[3:]))
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        anchored = policy.local_support_target(env.command, 0., box, world_reference=reference)
        legacy = policy.local_support_target(env.command, 0., box)
        for actual, expected in zip(self.kin.forward(anchored[:15]), reference):
            np.testing.assert_allclose((self.transform(second) @ actual)[:3, 3], expected[:3, 3], atol=2e-5)
        legacy_error = max(np.linalg.norm((self.transform(second) @ actual)[:3, 3] - expected[:3, 3])
                           for actual, expected in zip(self.kin.forward(legacy[:15]), reference))
        self.assertGreater(legacy_error, .0004)

    def test_invalid_world_reference_is_rejected(self):
        env, _, _, box, _ = self.environment()
        for reference in (np.zeros((1, 4, 4)), np.full((2, 4, 4), np.nan)):
            with self.subTest(shape=reference.shape), self.assertRaisesRegex(RuntimeError, 'INVALID_OPEN_SUPPORT_REFERENCE'):
                SeatEdgeArmExecutePolicy(env, 'grasp').local_support_target(env.command, .001, box, world_reference=reference)

    def feedback(self, discard_world_reference):
        env, poses, measured, box, _ = self.environment()
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        sequence = self.data['base_pose_sequence']
        state = {'index': 0, 'poses': poses, 'measured': measured}
        env.base.get_world_pose = lambda: (np.asarray(sequence[state['index']][:3]), np.asarray(sequence[state['index']][3:]))
        names = self.kin.names + [f'{side}_gripper_finger_joint{i}' for side in ('left', 'right') for i in (1, 2)]
        nominal = self.model.forward(dict(zip(names, env.command)), self.transform(sequence[0]))
        residuals = {name: np.linalg.inv(nominal[name]) @ pose for name, pose in poses.items()}
        joint_residual = measured - env.command
        policy.measured_calibration_state = Mock(side_effect=lambda: (state['poses'], state['measured'], box))
        if discard_world_reference:
            target_function = policy.local_support_target
            policy.local_support_target = lambda initial, correction, box, **ignored: target_function(initial, correction, box)
        def motion(target, duration, phase):
            self.assertEqual((duration, phase), (.8, 'approach'))
            state['index'] += 1
            env.command = target.copy()
            nominal = self.model.forward(dict(zip(names, target)), self.transform(sequence[state['index']]))
            state['poses'] = {name: nominal[name] @ residual for name, residual in residuals.items()}
            state['measured'] = target + joint_residual
        policy.interpolate = Mock(side_effect=motion)
        error = None
        try:
            policy.center_support_open()
        except RuntimeError as exc:
            error = str(exc)
        final = policy.support_alignment(state['poses'], state['measured'], box)['support_x_error_m']
        return error, final, policy.interpolate.call_count, env

    def test_recorded_drift_exposes_legacy_failure_and_world_reference_converges(self):
        old_error, old_final, old_calls, _ = self.feedback(True)
        self.assertEqual(old_error, 'OPEN_SUPPORT_ALIGNMENT_NOT_CONVERGED')
        self.assertEqual(old_calls, 4)
        self.assertGreater(abs(old_final), .00005)
        error, final, calls, env = self.feedback(False)
        self.assertIsNone(error)
        self.assertLessEqual(abs(final), .00005)
        self.assertLessEqual(calls, 4)
        self.assertLessEqual(abs(env.edge_grasp_x_offset), .02)


if __name__ == '__main__':
    unittest.main()
