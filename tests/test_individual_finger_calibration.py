from catalog_test_assets import robot_urdf
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy, BaseMovePolicy, SeatEdgeArmExecutePolicy, SeatEdgeArmPlanPolicy
from grutopia_extension.interaction_clearance.task import ClearanceTask, yaw_rotation
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualUrdfKinematics


class IndividualFingerCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).parents[1]
        cls.fixture=json.loads((Path(__file__).parent/'fixtures/native_pre_squeeze.json').read_text())
        urdf=robot_urdf()
        cls.model=RobotCollisionModel(urdf)
        cls.kinematics=BimanualUrdfKinematics(urdf)

    @staticmethod
    def transform(position,quaternion):
        value=np.eye(4);value[:3,3]=position;value[:3,:3]=Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix();return value

    def environment(self):
        data=self.fixture;task=ClearanceTask.from_dict(data['task']);o=data['observation'];sample=data['sample']
        env=SimpleNamespace(task=task,box=SimpleNamespace(dimensions=task.dimensions),collision_model=self.model,bimanual=self.kinematics,
            grasp_mode='stool_seat_edge',command=np.asarray(data['command']).copy(),control_indices=np.arange(19),recorder=Mock(),
            rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),object_bottom=lambda:.0001,
            base=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['base_pose_world'][:3]),np.asarray(o['base_pose_world'][3:]))),
            object=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['object_pose_world'][:3]),np.asarray(o['object_pose_world'][3:]))),
            finger_force_vectors=np.zeros((4,3)),robot=SimpleNamespace(get_joint_positions=Mock(return_value=np.asarray(o['joint_positions']))))
        env.wall_objects={w.name:self.model.fcl.CollisionObject(self.model.fcl.Box(*w.dimensions),self.model.fcl.Transform(yaw_rotation(w.yaw_rad),w.position)) for w in task.scene_obstacles}
        poses={n:self.transform(p,q) for n,p,q in zip(data['link_names'],sample['link_positions'],sample['link_quaternions_wxyz'])}
        box=self.transform(sample['object_position'],sample['object_quaternion_wxyz']);box[:3,3]+=box[:3,:3]@np.array((0.,0.,task.dimensions[2]/2))
        return env,poses,np.asarray(o['joint_positions']),box

    def test_native_asymmetry_produces_distinct_bounded_finger_targets(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        targets,details=policy.finger_touch_targets(poses,measured,box)
        np.testing.assert_allclose(targets,(.03231848,-.03515547,.03784617,-.03490580),atol=1e-6)
        self.assertGreater(np.ptp(np.abs(targets)),.005)
        self.assertTrue(all(0<d['touch_aperture_m']-abs(d['command_q_m'])<=.000900001 for d in details))

    def test_touch_calibration_is_invariant_to_world_pose(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        expected,_=policy.finger_touch_targets(poses,measured,box)
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',(.1,-.2,.8)).as_matrix();world[:3,3]=(.6,.2,.3)
        actual,_=policy.finger_touch_targets({n:world@p for n,p in poses.items()},measured,world@box)
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_real_native_calibration_path_keeps_original_safety_margins(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        finger,_=policy.finger_touch_targets(poses,measured,box);target=env.command.copy();target[15:]=finger
        audit=policy.certify_calibration(poses,measured,target,box)
        self.assertEqual(len(audit),101)
        self.assertGreaterEqual(min(r['minimum_noncontact_clearance_m'] for r in audit),.015)
        self.assertGreaterEqual(min(r['minimum_wall_clearance_m'] for r in audit),.08)
        self.assertGreaterEqual(min(r['minimum_finger_gap_m'] for r in audit),-.002)

    def test_grasp_reference_preserves_commands_not_measured_deflection(self):
        env,poses,measured,box=self.environment();env.robot.get_joint_positions=Mock(side_effect=AssertionError('must preserve commands'))
        policy=SeatEdgeArmExecutePolicy(env,'grasp');policy.freeze_commanded_grasp()
        origin=box.copy();origin[:3,3]-=origin[:3,:3]@np.array((0.,0.,env.task.dimensions[2]/2))
        p,q=env.base.get_world_pose();base=self.transform(p,q)
        for reference,tool in zip(env.edge_commanded_grasp_frames,env.bimanual.forward(env.command[:15])):
            np.testing.assert_allclose(origin@reference,base@tool,atol=1e-10)
        np.testing.assert_array_equal(env.edge_finger_command,env.command[15:])
        env.robot.get_joint_positions.assert_not_called()

    def test_preserved_grasp_passes_accuracy_check_without_nominal_recentering(self):
        env,poses,measured,box=self.environment();SeatEdgeArmExecutePolicy(env,'grasp').freeze_commanded_grasp()
        p,q=env.base.get_world_pose()
        details=SeatEdgeArmPlanPolicy(env).check_lift_grip(env.command,self.transform(p,q),box)
        self.assertLess(details['grip_span_error_m'],1e-10)

    def test_invalid_saved_finger_command_is_rejected(self):
        env,_,_,_=self.environment()
        for command in ([0.]*4,[.03,.03,.03,.03],[float('nan'),-.03,.03,-.03],[.03,-.03]):
            env.edge_finger_command=command
            with self.assertRaisesRegex(RuntimeError,'INVALID_CALIBRATED'):
                SeatEdgeArmPlanPolicy(env).finger_targets(False)

    def loop_fixture(self,contacts):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box))
        desired=np.array((.032,-.035,.038,-.034))
        policy.finger_touch_targets=Mock(return_value=(desired,[]))
        policy.certify_calibration=Mock(return_value=[{'minimum_noncontact_clearance_m':.04,'minimum_wall_clearance_m':.18}])
        policy.freeze_commanded_grasp=Mock()
        def motion(target,duration,phase):env.command=target.copy()
        policy.interpolate=Mock(side_effect=motion)
        if contacts:
            rotation=env.rotation(env.object.get_world_pose()[1])
            env.finger_force_vectors=np.array(((0,0,-2),(0,0,2),(0,0,2),(0,0,-2)))@rotation.T
        return env,policy,desired

    def test_missing_contacts_exhaust_bounded_calibration_without_lift(self):
        env,policy,_=self.loop_fixture(False);previous=env.command[15:].copy();signs=np.array((1,-1,1,-1))
        with self.assertRaisesRegex(RuntimeError,'FOUR_FINGER_CALIBRATION_NOT_CONVERGED'):policy.calibrate_contact()
        self.assertEqual(policy.interpolate.call_count,12)
        for call in policy.interpolate.call_args_list:
            current=call.args[0][15:]
            self.assertLessEqual(max(signs*(previous-current)),.00200001);previous=current
            self.assertEqual(call.args[1:],(.4,'squeeze'))
        policy.freeze_commanded_grasp.assert_not_called()

    def test_four_contacts_require_two_converged_checks_before_freezing(self):
        env,policy,desired=self.loop_fixture(True);env.command[15:]=desired
        policy.calibrate_contact()
        self.assertEqual(policy.interpolate.call_count,2)
        policy.freeze_commanded_grasp.assert_called_once()

    def test_command_tracking_residual_reconstructs_all_native_finger_poses(self):
        env,poses,measured,box=self.environment()
        SeatEdgeArmExecutePolicy(env,'grasp').freeze_commanded_grasp()
        sample=self.fixture['sample'];env.collision_link_names=self.fixture['link_names']
        env.collision_links=SimpleNamespace(get_world_poses=lambda:(sample['link_positions'],sample['link_quaternions_wxyz']))
        SeatEdgeArmPlanPolicy(env).refresh_finger_tracking()
        p,q=env.base.get_world_pose();names=env.bimanual.names+[f'{s}_gripper_finger_joint{i}' for s in ('left','right') for i in (1,2)]
        nominal=env.collision_model.forward(dict(zip(names,env.command)),self.transform(p,q))
        self.assertEqual(len(env.edge_finger_tracking_residuals),4)
        for name,residual in env.edge_finger_tracking_residuals.items():
            np.testing.assert_allclose(nominal[name]@residual,poses[name],atol=1e-10)

    def test_placement_uses_calibrated_edge_planner_and_commanded_initial_state(self):
        env,_,_,_=self.environment();env.edge_commanded_grasp_frames=(np.eye(4),np.eye(4))
        checker=SimpleNamespace(relative_box=np.eye(4),details=lambda pose:{'accepted':True,'minimum_wall_clearance_m':.20})
        policy=BaseMovePolicy(env)
        with patch('grutopia_extension.interaction_clearance.primitives.FrozenBaseCollisionChecker',return_value=checker), \
             patch('grutopia_extension.interaction_clearance.primitives.placement_candidates',return_value=[np.array((.8,2.5))]), \
             patch.object(ArmPlanPolicy,'targets',side_effect=AssertionError('legacy side planner used')), \
             patch.object(SeatEdgeArmPlanPolicy,'targets',return_value=([np.zeros(19)],['release'],[4.])) as targets, \
             patch.object(SeatEdgeArmPlanPolicy,'certify',return_value=[{'minimum_wall_clearance_m':.2}]) as certify, \
             patch.object(policy,'plan',return_value=[np.zeros(3),np.array((0.,.7,0.))]):
            policy.select_placement()
        targets.assert_called_once();certify.assert_called_once()
        np.testing.assert_array_equal(targets.call_args.args[2],env.command)
        np.testing.assert_array_equal(certify.call_args.args[1],env.command)


if __name__=='__main__':unittest.main()
