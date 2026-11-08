from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from scipy.spatial.transform import Rotation

import test_individual_finger_calibration as fixtures
from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy,SeatEdgeArmExecutePolicy,SeatEdgeArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask


class WorldLevelLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.IndividualFingerCalibrationTests.setUpClass()

    def prepared(self):
        env,_,_,_=fixtures.IndividualFingerCalibrationTests().environment()
        policy=SeatEdgeArmExecutePolicy(env,'grasp');policy.freeze_commanded_grasp();policy.capture_retention_reference()
        return env,policy

    def test_measured_pitch_is_not_copied_into_next_target(self):
        env,policy=self.prepared();origin=env.object.get_world_pose()[0]
        level=env.edge_world_lift_rotation.copy()
        first,rotation=policy.lift_pose_target(.22,origin,level,env.command)
        tipped=level@Rotation.from_euler('y',.04).as_matrix()
        shifted=origin+np.array((.01,-.003,.006))
        second,after=policy.lift_pose_target(.22,shifted,tipped,env.command)
        np.testing.assert_allclose(after,level,atol=1e-12)
        np.testing.assert_allclose(second,first,atol=1e-12)

    def test_xy_anchor_preserves_commanded_hand_center_without_object_writes(self):
        env,policy=self.prepared();env.object.set_world_pose=Mock(side_effect=AssertionError('object must remain dynamic'))
        center=env.object.get_world_pose()[0]
        target,rotation=policy.lift_pose_target(.2201,center,env.edge_world_lift_rotation@Rotation.from_euler('y',-.04).as_matrix(),env.command)
        p,q=env.base.get_world_pose();base=np.eye(4);base[:3,3]=p;base[:3,:3]=env.rotation(q)
        tools=[base@t for t in env.bimanual.forward(env.command[:15])]
        actual_midpoint=(tools[0][:3,3]+tools[1][:3,3])/2
        local_midpoint=(env.edge_commanded_grasp_frames[0][:3,3]+env.edge_commanded_grasp_frames[1][:3,3])/2
        np.testing.assert_allclose((target+rotation@local_midpoint)[:2],actual_midpoint[:2],atol=1e-12)
        self.assertEqual(target[2],.2201);env.object.set_world_pose.assert_not_called()

    def test_reference_is_gravity_level_even_if_measured_grip_is_slightly_tilted(self):
        env,policy=self.prepared();p,q=env.object.get_world_pose();r=env.rotation(q)@Rotation.from_euler('y',.03).as_matrix();quaternion=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
        env.object.get_world_pose=lambda:(p,quaternion)
        policy.capture_retention_reference()
        np.testing.assert_allclose(env.edge_world_lift_rotation[:,2],(0.,0.,1.),atol=1e-12)

    def test_pose_certificate_interpolates_object_origin_and_orientation_continuously(self):
        task=ClearanceTask();env=SimpleNamespace(task=task);seen=[]
        yaw=.3;level=Rotation.from_euler('z',yaw).as_matrix();initial_r=level@Rotation.from_euler('y',.04).as_matrix()
        origin=np.array((.72,.02,.015));target=np.array((.715,.025,.04));half=np.array((0.,0.,task.dimensions[2]/2))
        box=np.eye(4);box[:3,:3]=initial_r;box[:3,3]=origin+initial_r@half
        def capture(joints,base,pose):
            seen.append(pose.copy());return {'minimum_noncontact_clearance_m':.05,'minimum_wall_clearance_m':.1}
        planner=ArmPlanPolicy(env)
        with patch.object(planner,'configuration',side_effect=capture):
            planner.certify('grasp',np.zeros(19),[np.ones(19)],['lift'],np.eye(4),origin,yaw,box,
                            lift_distance=.025,lift_origin_target=target,lift_rotation_target=level)
        self.assertEqual(len(seen),101);np.testing.assert_allclose(seen[0],box,atol=1e-12)
        np.testing.assert_allclose(seen[-1][:3,:3],level,atol=1e-12)
        np.testing.assert_allclose(seen[-1][:3,3],target+level@half,atol=1e-12)
        middle_r=level@Rotation.from_euler('y',.02).as_matrix()
        np.testing.assert_allclose(seen[50][:3,:3],middle_r,atol=1e-12)
        np.testing.assert_allclose(seen[50][:3,3],(origin+target)/2+middle_r@half,atol=1e-12)

    def test_uncalibrated_fallback_keeps_existing_measured_pose_behavior(self):
        policy=SeatEdgeArmExecutePolicy(SimpleNamespace(),'grasp')
        origin=np.array((.7,.1,.02));rotation=Rotation.from_euler('y',.02).as_matrix()
        target,actual=policy.lift_pose_target(.04,origin,rotation,np.zeros(19))
        np.testing.assert_array_equal(target,(.7,.1,.04));np.testing.assert_array_equal(actual,rotation)

    def test_loaded_command_refresh_preserves_retention_baseline_and_commands(self):
        env,policy=self.prepared();native=tuple(frame.copy() for frame in env.edge_native_grasp_frames);command=env.command.copy()
        calibrated_rotations=tuple(frame[:3,:3].copy() for frame in env.edge_commanded_grasp_frames)
        p,q=env.object.get_world_pose();p=p+np.array((.001,0.,-.002));rotation=env.rotation(q)@Rotation.from_euler('y',.01).as_matrix();quaternion=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        env.object.get_world_pose=lambda:(p,quaternion)
        SeatEdgeArmPlanPolicy(env).refresh_commanded_grasp_reference()
        origin=np.eye(4);origin[:3,3]=p;origin[:3,:3]=rotation
        bp,bq=env.base.get_world_pose();base=np.eye(4);base[:3,3]=bp;base[:3,:3]=env.rotation(bq)
        for reference,tool,fixed_rotation in zip(env.edge_commanded_grasp_frames,env.bimanual.forward(command[:15]),calibrated_rotations):
            np.testing.assert_allclose((origin@reference)[:3,3],(base@tool)[:3,3],atol=1e-10)
            np.testing.assert_array_equal(reference[:3,:3],fixed_rotation)
        for expected,actual in zip(native,env.edge_native_grasp_frames):np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(env.command,command)


if __name__=='__main__':unittest.main()
