"""Recorded support-alignment failure, strict geometry gates and bounded feedback."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_individual_finger_calibration as fixtures
from grutopia_extension.interaction_clearance.primitives import SeatEdgeArmExecutePolicy, SeatEdgeArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask, yaw_rotation


class LocalSupportCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IndividualFingerCalibrationTests.setUpClass()
        cls.model = fixtures.IndividualFingerCalibrationTests.model
        cls.kin = fixtures.IndividualFingerCalibrationTests.kinematics

    def environment(self, index=0, data=None):
        path = Path(__file__).parent / 'fixtures/open_support_alignment' / f'alignment_state_{index:02d}.json'
        if data is None:
            data = json.loads(path.read_text())
        task = ClearanceTask.from_dict(data['task'])
        observation, sample = data['observation'], data['sample']
        transform = fixtures.IndividualFingerCalibrationTests.transform
        measured = np.asarray(observation['joint_positions'])
        env = SimpleNamespace(task=task, box=SimpleNamespace(dimensions=task.dimensions),
            collision_model=self.model, bimanual=self.kin, control_indices=np.arange(19),
            command=np.asarray(data['command']), recorder=Mock(),
            rotation=lambda q: Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix(),
            object_bottom=lambda: data['object_bottom_m'], edge_grasp_x_offset=data['edge_grasp_x_offset'],
            edge_support_open_target_x=data['recorded_probe']['open_support_target_x_m'],
            base=SimpleNamespace(get_world_pose=lambda: (np.asarray(observation['base_pose_world'][:3]), np.asarray(observation['base_pose_world'][3:]))),
            object=SimpleNamespace(get_world_pose=lambda: (np.asarray(observation['object_pose_world'][:3]), np.asarray(observation['object_pose_world'][3:]))))
        env.wall_objects = {wall.name: self.model.fcl.CollisionObject(self.model.fcl.Box(*wall.dimensions),
            self.model.fcl.Transform(yaw_rotation(wall.yaw_rad), wall.position)) for wall in task.scene_obstacles}
        poses = {name: transform(p, q) for name, p, q in zip(data['link_names'], sample['link_positions'], sample['link_quaternions_wxyz'])}
        box = transform(sample['object_position'], sample['object_quaternion_wxyz'])
        box[:3, 3] += box[:3, :3] @ np.array((0., 0., task.dimensions[2] / 2))
        return env, poses, measured, box, data

    def test_legacy_world_native_probes_match_original_failure(self):
        for index in (0, 4):
            with self.subTest(index=index):
                env, poses, measured, box, data = self.environment(index)
                error = SeatEdgeArmExecutePolicy(env, 'grasp').support_alignment(poses, measured, box, object_frame=False)['support_x_error_m']
                self.assertAlmostEqual(error, data['recorded_probe']['support_x_error_m'], delta=1e-7)
                self.assertGreater(abs(error), .004)

    def test_local_translation_preserves_commanded_orientation_and_pose_precision(self):
        env, _, _, box, _ = self.environment()
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        initial = env.command.copy()
        expected_delta = env.rotation(env.base.get_world_pose()[1]).T @ box[:3, 0] * .004
        target = policy.local_support_target(initial, .004, box)
        for before, after in zip(self.kin.forward(initial[:15]), self.kin.forward(target[:15])):
            np.testing.assert_allclose(after[:3, 3] - before[:3, 3], expected_delta, atol=2e-5)
            self.assertLess(np.linalg.norm(Rotation.from_matrix(before[:3, :3] @ after[:3, :3].T).as_rotvec()), .0001)
        np.testing.assert_array_equal(env.command, initial)
        np.testing.assert_allclose(target[15:], (.045, -.045, .045, -.045))

    def test_generic_ik_acceptance_is_not_enough_for_local_correction(self):
        env, _, _, box, _ = self.environment()
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        with patch.object(self.kin, 'solve', return_value=env.command[:15].copy()):
            with self.assertRaisesRegex(RuntimeError, 'LOCAL_IK_INACCURATE'):
                policy.local_support_target(env.command, .004, box)

    def test_invalid_corrections_are_rejected(self):
        env, _, _, box, _ = self.environment()
        for correction in (float('nan'), float('inf'), .01001, -.01001):
            with self.subTest(correction=correction), self.assertRaisesRegex(RuntimeError, 'INVALID_OPEN_SUPPORT'):
                SeatEdgeArmExecutePolicy(env, 'grasp').local_support_target(env.command, correction, box)

    def test_original_nominal_collision_gate_cannot_be_bypassed(self):
        env, poses, _, box, _ = self.environment()
        policy = SeatEdgeArmExecutePolicy(env, 'grasp')
        with patch.object(SeatEdgeArmPlanPolicy, 'certify', side_effect=RuntimeError('nominal_rejection')):
            with self.assertRaisesRegex(RuntimeError, 'nominal_rejection'):
                policy.certify_support_translation(env.command, env.command, poses, box)

    def test_measured_body_environment_collision_is_rejected(self):
        env, poses, _, box, _ = self.environment()
        poses['left_arm_link3'][:3, 3] = env.task.scene_obstacles[0].position
        with self.assertRaises(RuntimeError):
            SeatEdgeArmExecutePolicy(env, 'grasp').certify_support_translation(env.command, env.command, poses, box)

    def test_missing_native_body_geometry_is_rejected(self):
        env, poses, _, box, _ = self.environment()
        poses.pop('left_arm_link3')
        with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE_NATIVE_SUPPORT'):
            SeatEdgeArmExecutePolicy(env, 'grasp').certify_support_translation(env.command, env.command, poses, box)

    def test_all_recorded_native_states_converge_with_four_numerical_updates(self):
        # Tracking offsets are held constant in this numerical test. Three
        # intermediate fixtures have documented online-nearest-point differences;
        # this test uses their recorded poses, not a claim of exact online replay.
        for index in range(5):
            with self.subTest(index=index):
                env, poses, measured, box, _ = self.environment(index)
                policy = SeatEdgeArmExecutePolicy(env, 'grasp')
                names = self.kin.names + [f'{side}_gripper_finger_joint{i}' for side in ('left', 'right') for i in (1, 2)]
                base = fixtures.IndividualFingerCalibrationTests.transform(*env.base.get_world_pose())
                original = self.model.forward(dict(zip(names, env.command)), base)
                residuals = {name: np.linalg.inv(original[name]) @ pose for name, pose in poses.items()}
                joint_residual = measured - env.command
                state = {'poses': poses, 'measured': measured}
                policy.measured_calibration_state = Mock(side_effect=lambda: (state['poses'], state['measured'], box))
                def motion(target, duration, phase):
                    self.assertEqual((duration, phase), (.8, 'approach'))
                    env.command = target.copy()
                    nominal = self.model.forward(dict(zip(names, target)), base)
                    state['poses'] = {name: nominal[name] @ residual for name, residual in residuals.items()}
                    state['measured'] = target + joint_residual
                policy.interpolate = Mock(side_effect=motion)
                policy.center_support_open()
                self.assertLessEqual(policy.interpolate.call_count, 4)
                self.assertLessEqual(abs(env.edge_grasp_x_offset), .02)
                final = policy.support_alignment(state['poses'], state['measured'], box)['support_x_error_m']
                self.assertLessEqual(abs(final - env.edge_support_open_target_x), .00005)
                for call in env.recorder.event.call_args_list:
                    if call.args[0] == 'support_alignment_motion':
                        self.assertEqual(call.args[1]['samples'], 202)
                        self.assertGreaterEqual(call.args[1]['minimum_body_gap_m'], .015)
                        self.assertGreaterEqual(call.args[1]['minimum_wall_gap_m'], .08)


if __name__ == '__main__':
    unittest.main()
