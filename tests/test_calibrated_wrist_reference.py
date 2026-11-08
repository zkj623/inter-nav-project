import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import test_individual_finger_calibration as fixtures
from grutopia_extension.interaction_clearance.primitives import SeatEdgeArmPlanPolicy, SeatEdgeArmExecutePolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask


class CalibratedWristReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IndividualFingerCalibrationTests.setUpClass()
        cls.kinematics = fixtures.IndividualFingerCalibrationTests.kinematics
        cls.data = json.loads((Path(__file__).parent/'fixtures/r33_wrist_rotation.json').read_text())

    def recorded(self, data, calibrated=True):
        o=data['observation']
        transform=fixtures.IndividualFingerCalibrationTests.transform
        origin=transform(o['object_pose_world'][:3],o['object_pose_world'][3:])
        base=transform(o['base_pose_world'][:3],o['base_pose_world'][3:])
        original=tuple(np.asarray(t).copy() for t in data['original_commanded_reference']['tool_frames_in_object'])
        task=ClearanceTask.from_dict(data['task'])
        env=SimpleNamespace(task=task,box=SimpleNamespace(dimensions=task.dimensions),bimanual=self.kinematics,
            command=np.asarray(data['command']).copy(),control_indices=np.arange(19),recorder=Mock(),
            rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),
            object=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['object_pose_world'][:3]),np.asarray(o['object_pose_world'][3:]))),
            base=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['base_pose_world'][:3]),np.asarray(o['base_pose_world'][3:]))),
            edge_commanded_grasp_frames=original,
            edge_native_grasp_frames=tuple(np.asarray(t).copy() for t in data['native_reference']['tool_frames_in_object']))
        if calibrated:
            env.edge_calibrated_grasp_rotations=tuple(t[:3,:3].copy() for t in original)
        return env,origin,base,original

    def test_recorded_rotation_drift_is_removed_without_reanchoring_positions_or_retention(self):
        legacy_max=0.
        for data in self.data['states']:
            env,origin,base,original=self.recorded(data)
            command=env.command.copy();native=tuple(t.copy() for t in env.edge_native_grasp_frames)
            expected=tuple(np.linalg.inv(origin)@base@t for t in env.bimanual.forward(command[:15]))
            for _ in range(3):
                SeatEdgeArmPlanPolicy(env).refresh_commanded_grasp_reference()
                for actual,position_reference,rotation_reference in zip(env.edge_commanded_grasp_frames,expected,original):
                    np.testing.assert_allclose(actual[:3,3],position_reference[:3,3],atol=1e-12)
                    np.testing.assert_array_equal(actual[:3,:3],rotation_reference[:3,:3])
                np.testing.assert_array_equal(env.command,command)
                for actual,prior in zip(env.edge_native_grasp_frames,native):np.testing.assert_array_equal(actual,prior)
            legacy,_,_,_=self.recorded(data,calibrated=False)
            SeatEdgeArmPlanPolicy(legacy).refresh_commanded_grasp_reference()
            legacy_max=max(legacy_max,max(float(Rotation.from_matrix(t[:3,:3]@r[:3,:3].T).magnitude())
                for t,r in zip(legacy.edge_commanded_grasp_frames,original)))
        self.assertGreater(legacy_max,.04)

    def test_new_ground_calibration_replaces_each_wrist_reference(self):
        env,_,_,_=fixtures.IndividualFingerCalibrationTests().environment()
        policy=SeatEdgeArmExecutePolicy(env,'grasp');policy.freeze_commanded_grasp()
        old=tuple(r.copy() for r in env.edge_calibrated_grasp_rotations)
        p,q=env.object.get_world_pose();changed=env.rotation(q)@Rotation.from_euler('y',.02).as_matrix()
        env.object.get_world_pose=lambda:(p,Rotation.from_matrix(changed).as_quat()[[3,0,1,2]])
        policy.freeze_commanded_grasp()
        for reference,frame,previous in zip(env.edge_calibrated_grasp_rotations,env.edge_commanded_grasp_frames,old):
            np.testing.assert_array_equal(reference,frame[:3,:3])
            self.assertGreater(Rotation.from_matrix(reference@previous.T).magnitude(),.019)
            self.assertFalse(np.shares_memory(reference,frame))

    def test_reopening_invalidates_calibrated_rotation_before_another_grasp(self):
        env,poses,measured,box=fixtures.IndividualFingerCalibrationTests().environment()
        policy=SeatEdgeArmExecutePolicy(env,'grasp');policy.freeze_commanded_grasp();policy.capture_retention_reference()
        with patch.object(policy,'measured_calibration_state',return_value=(poses,measured,box)),\
             patch.object(policy,'certify_calibration'),patch.object(policy,'interpolate'):
            policy.reopen_for_alignment()
        self.assertFalse(hasattr(env,'edge_calibrated_grasp_rotations'))
        self.assertFalse(hasattr(env,'edge_native_grasp_frames'))
        self.assertFalse(hasattr(env,'edge_commanded_grasp_frames'))


if __name__=='__main__':unittest.main()
