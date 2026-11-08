from catalog_test_assets import robot_urdf
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import grasp_contact_established, grasp_contact_retained
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualUrdfKinematics


class SupportedRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).parents[1]
        cls.data=json.loads((Path(__file__).parent/'fixtures/native_loaded_retention.json').read_text())
        urdf=robot_urdf();cls.model=RobotCollisionModel(urdf);cls.kin=BimanualUrdfKinematics(urdf)

    @staticmethod
    def transform(v):
        result=np.eye(4);result[:3,3]=v[:3];result[:3,:3]=Rotation.from_quat(np.asarray(v[3:])[[1,2,3,0]]).as_matrix();return result

    def environment(self):
        d=self.data;o=d['observation'];before=d['reference_observation'];sample=d['sample']
        task=ClearanceTask.from_dict(d['task'])
        env=SimpleNamespace(task=task,box=SimpleNamespace(dimensions=task.dimensions),collision_model=self.model,bimanual=self.kin,
            grasp_mode='stool_seat_edge',control_indices=np.arange(19),recorder=Mock(),
            finger_force_vectors=np.asarray(o['finger_normal_force_vectors']).copy(),
            native_positions=np.asarray(sample['link_positions']).copy(),native_quaternions=np.asarray(sample['link_quaternions_wxyz']).copy(),
            object_pose=np.asarray(o['object_pose_world']).copy(),angular=np.asarray(o['object_angular_velocity']).copy(),
            rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),
            base=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['base_pose_world'][:3]),np.asarray(o['base_pose_world'][3:]))),
            robot=SimpleNamespace(get_joint_positions=lambda:np.asarray(o['joint_positions'])),collision_link_names=d['link_names'])
        env.object=SimpleNamespace(get_world_pose=lambda:(env.object_pose[:3],env.object_pose[3:]),get_angular_velocity=lambda:env.angular)
        env.collision_links=SimpleNamespace(get_world_poses=lambda:(env.native_positions,env.native_quaternions))
        env.edge_native_grasp_frames=tuple(np.linalg.inv(self.transform(before['object_pose_world']))@self.transform(before['base_pose_world'])@t
            for t in self.kin.forward(np.asarray(before['joint_positions'])[:15]))
        return env

    def test_loaded_native_state_requires_support_geometry_not_uniform_pressure(self):
        env=self.environment();before=self.data['reference_observation']
        rotation=self.transform(before['object_pose_world'])[:3,:3]
        self.assertTrue(np.all((np.asarray(before['finger_normal_force_vectors'])@rotation[:,2])*[-1,1,1,-1]>.5))
        self.assertFalse(grasp_contact_established(env))
        self.assertTrue(grasp_contact_retained(env))
        self.assertEqual(env.grasp_retention_diagnostics['reason'],'supported_and_geometrically_retained')

    def test_initial_establishment_still_requires_four_loaded_fingers(self):
        env=self.environment();del env.edge_native_grasp_frames
        self.assertFalse(grasp_contact_retained(env))

    def test_loss_of_either_lower_support_is_rejected(self):
        for index in (0,3):
            env=self.environment();env.finger_force_vectors[index]=0.
            self.assertFalse(grasp_contact_retained(env))

    def test_oppositely_loaded_upper_finger_is_rejected(self):
        env=self.environment();env.finger_force_vectors[1]=-env.rotation(env.object_pose[3:])[:,2]
        self.assertFalse(grasp_contact_retained(env))

    def test_upper_finger_clearance_above_one_mm_is_rejected(self):
        env=self.environment();index=env.collision_link_names.index('left_gripper_finger_link2')
        env.native_positions[index]+=env.rotation(env.object_pose[3:])[:,2]*.002
        self.assertFalse(grasp_contact_retained(env))
        self.assertEqual(env.grasp_retention_diagnostics['reason'],'finger_outside_retention_band')

    def test_tangential_grasp_slip_is_rejected(self):
        env=self.environment();env.edge_native_grasp_frames[0][0,3]-=.004
        self.assertFalse(grasp_contact_retained(env))
        self.assertEqual(env.grasp_retention_diagnostics['reason'],'grasp_slip')

    def test_relative_grasp_rotation_is_rejected(self):
        env=self.environment();env.edge_native_grasp_frames[0][:3,:3] @= Rotation.from_euler('z',.07).as_matrix()
        self.assertFalse(grasp_contact_retained(env))

    def test_absolute_tilt_above_five_degrees_is_rejected(self):
        env=self.environment();rotation=Rotation.from_euler('x',.12).as_matrix()@env.rotation(env.object_pose[3:])
        env.object_pose[3:]=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        self.assertFalse(grasp_contact_retained(env))
        self.assertEqual(env.grasp_retention_diagnostics['reason'],'object_tilt')

    def test_excessive_object_angular_speed_is_rejected(self):
        env=self.environment();env.angular=np.array((.36,0.,0.))
        self.assertFalse(grasp_contact_retained(env))
        self.assertEqual(env.grasp_retention_diagnostics['reason'],'object_angular_speed')

    def test_nonfinite_forces_cannot_establish_retention(self):
        env=self.environment();env.finger_force_vectors[0,0]=float('nan')
        self.assertFalse(grasp_contact_retained(env))


if __name__=='__main__':unittest.main()
