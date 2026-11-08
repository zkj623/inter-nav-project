from catalog_test_assets import robot_urdf
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd

from grutopia_extension.interaction_clearance import primitives as p
from grutopia_extension.interaction_clearance.task import ClearanceTask,yaw_rotation
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualUrdfKinematics


class CapturedReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).parent
        cls.capture=json.loads((folder/'fixtures/r34_native_release_call.json').read_text())
        cls.model=RobotCollisionModel(robot_urdf())
        cls.kin=BimanualUrdfKinematics(robot_urdf())
        cls.model.verify_stool_stage(Usd.Stage.Open(str(folder/'fixtures/r34_release_scene.usda')),cls.capture['task']['dimensions'])

    def environment(self):
        d=self.capture;o=d['observation'];task=ClearanceTask.from_dict(d['task'])
        env=SimpleNamespace(task=task,box=SimpleNamespace(dimensions=task.dimensions),bimanual=self.kin,collision_model=self.model,
            grasp_mode='stool_seat_edge',recorder=None,rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),
            base=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['base_pose_world'][:3]),np.asarray(o['base_pose_world'][3:]))),
            object=SimpleNamespace(get_world_pose=lambda:(np.asarray(o['object_pose_world'][:3]),np.asarray(o['object_pose_world'][3:]))),
            robot=SimpleNamespace(dof_names=d['joint_names'],get_joint_positions=lambda:np.asarray(d['all_measured_joint_positions'])),
            command=np.asarray(d['all_commanded_joint_positions']),control_indices=np.asarray(d['control_indices']),
            initial_control=np.asarray(d['initial_control']),collision_link_names=d['native_link_names'],
            collision_links=SimpleNamespace(get_world_poses=lambda:(np.asarray(d['native_link_positions']),np.asarray(d['native_link_quaternions_wxyz']))))
        for name,value in d['grasp_fields'].items():
            setattr(env,name,{k:np.asarray(v) for k,v in value.items()} if isinstance(value,dict) else np.asarray(value))
        env.wall_objects={w.name:self.model.fcl.CollisionObject(self.model.fcl.Box(*w.dimensions),
            self.model.fcl.Transform(yaw_rotation(w.yaw_rad),w.position)) for w in task.scene_obstacles}
        return env

    def arguments(self):
        values=dict(self.capture['arguments'])
        for name in ('initial','base','center','current_box'):values[name]=np.asarray(values[name])
        values['targets']=[np.asarray(q) for q in values['targets']]
        return values

    def test_captured_original_release_still_fails_original_body_margin(self):
        arm=p.SeatEdgeArmPlanPolicy(self.environment())
        with self.assertRaises(p.InternalArmPathError) as failure:arm.certify(**self.arguments())
        self.assertAlmostEqual(failure.exception.details['minimum_noncontact_clearance_m'],.01469415615239817,places=8)
        self.assertGreater(failure.exception.details['minimum_wall_clearance_m'],.08)

    def test_staged_release_generates_safe_trajectory_and_preserves_opening_and_home(self):
        env=self.environment();arm=p.SeatEdgeArmPlanPolicy(env);args=self.arguments()
        targets,phases,durations=arm.targets('place',env.object.get_world_pose()[0],args['initial'],.07,release_strategy='staged')
        first=phases.index('release')
        np.testing.assert_array_equal(targets[first][:15],targets[first-1][:15])
        self.assertTrue(all(np.array_equal(q[15:],targets[first][15:]) for q in targets[first:-1]))
        np.testing.assert_array_equal(targets[-1],env.initial_control)
        self.assertEqual(phases[first:],['release','release','retract','retract','retract'])
        args.update(targets=targets,phases=phases)
        audit=arm.certify(**args)
        self.assertEqual(len(audit),1111)
        self.assertGreater(min(r['minimum_noncontact_clearance_m'] for r in audit),.04)
        self.assertGreaterEqual(min(r['minimum_wall_clearance_m'] for r in audit),.08)
        self.assertGreaterEqual(min(r['minimum_finger_gap_m'] for r in audit),-.002)
        self.assertTrue(all(t>0 for t in durations))

    def test_failing_internal_geometry_is_invariant_under_planar_world_transform(self):
        args=self.arguments();q=args['targets'][6]+.99*(args['targets'][7]-args['targets'][6])
        joints=dict(zip(self.kin.names,q[:15]))
        joints.update(zip((f'{s}_gripper_finger_joint{i}' for s in ('left','right') for i in (1,2)),q[15:]))
        box=args['current_box'].copy();box[:3,:3]=yaw_rotation(args['yaw']);box[:3,3]=np.r_[box[:2,3],self.capture['task']['dimensions'][2]/2+.0001]
        values=[]
        for angle,xy in ((0.,(0.,0.)),(.4,(1.2,-.7)),(-.6,(-.5,1.4))):
            world=np.eye(4);world[:3,:3]=yaw_rotation(angle);world[:2,3]=xy
            gaps=self.model.box_clearances(self.model.forward(joints,world@args['base']),world@box,self.capture['task']['dimensions'])
            values.append(min(gaps.values()))
        np.testing.assert_allclose(values,[.01469415615239817]*3,atol=1e-6,rtol=0)


class PlacementStrategyTests(unittest.TestCase):
    def exercise(self,internal):
        env=SimpleNamespace(task=ClearanceTask(object_kind='stool',dimensions=(.44,.52,.36)),box=SimpleNamespace(dimensions=(.44,.52,.36)),grasp_mode='stool_seat_edge',
            base=SimpleNamespace(get_world_pose=lambda:(np.zeros(3),np.array((1.,0.,0.,0.)))),
            object=SimpleNamespace(get_world_pose=lambda:(np.array((.7,0.,.2)),np.array((1.,0.,0.,0.)))),
            robot=SimpleNamespace(get_joint_positions=lambda:np.zeros(19)),control_indices=np.arange(19),
            rotation=lambda q:np.eye(3),recorder=SimpleNamespace(event=Mock()))
        attempts=[];calls=[]
        class Arm:
            def __init__(self,environment):self.strategy=None
            def targets(self,mode,center,initial,clearance,release_strategy=None):
                self.strategy=release_strategy;attempts.append(release_strategy)
                return [np.zeros(19)],['release'],[2.]
            def certify(self,*args):
                calls.append(self.strategy)
                if len(calls)==1:
                    if internal:raise p.InternalArmPathError({'minimum_noncontact_clearance_m':.014})
                    raise RuntimeError('environment obstacle')
        checker=SimpleNamespace(relative_box=np.eye(4),details=lambda target:{'accepted':True,'minimum_wall_clearance_m':.2})
        policy=p.BaseMovePolicy(env)
        with patch.object(p,'SeatEdgeArmPlanPolicy',Arm),patch.object(p,'FrozenBaseCollisionChecker',return_value=checker), \
             patch.object(p,'placement_candidates',return_value=[np.array((.8,1.)),np.array((.9,1.))]), \
             patch.object(policy,'plan',return_value=[np.zeros(3),np.ones(3)]):
            policy.select_placement()
        return attempts,calls,env

    def test_internal_rejection_switches_motion_before_repeating_placement(self):
        attempts,calls,env=self.exercise(True)
        self.assertEqual(attempts,['direct','staged']);self.assertEqual(calls,['direct','staged'])
        self.assertEqual(env.edge_release_strategy,'staged')

    def test_environment_rejection_keeps_searching_positions_with_same_motion(self):
        attempts,calls,env=self.exercise(False)
        self.assertEqual(attempts,['direct']);self.assertEqual(calls,['direct','direct'])
        self.assertEqual(env.edge_release_strategy,'direct')


if __name__=='__main__':unittest.main()
