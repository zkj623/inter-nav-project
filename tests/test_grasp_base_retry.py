from catalog_test_assets import robot_urdf
"""Bounded skill/base recovery before contact, with the unchanged physical gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import BaseMovePolicy, primitive_registry
from grutopia_extension.interaction_clearance.skill_runtime import Outcome, SkillContext, load_skill
from grutopia_extension.interaction_clearance.task import ClearanceTask, yaw_rotation
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox, BimanualUrdfKinematics
from grutopia_extension.manipulation.household_objects import grasp_height


class SkillBaseRetryTests(unittest.TestCase):
    def scenario(self, failures=0, kind='stool', return_home=False, blocked=False, lift_failed=False):
        calls = []
        attempts = [0]
        task = ClearanceTask(object_kind=kind, return_to_start=return_home)

        class Policy:
            def __init__(self, name): self.name = name
            def execute(self, task, **options):
                calls.append((self.name, options))
                if self.name == 'arm_plan' and options['mode'] == 'grasp':
                    attempts[0] += 1
                    if attempts[0] <= failures:
                        return Outcome(False, 'ARM_PLAN_REJECTED', {'retry_safe': True})
                if blocked and self.name == 'base_move' and 'grasp_distance_offset_m' in options:
                    return Outcome(False, 'BASE_PLAN_REJECTED', {'retry_safe': True})
                if lift_failed and self.name == 'grasp_lift':
                    return Outcome(False, 'ARM_EXECUTION_FAILURE', {'error': 'GRASP_LOST_DURING_LIFT'})
                return Outcome(True, 'DONE')

        ctx = SkillContext(task, {n:Policy(n) for n in ('base_move', 'arm_plan', 'grasp_lift', 'place_release')},
                           lambda:Outcome(True, 'ROAD_CLEARED'), lambda:{}, lambda *args:None)
        return ctx, calls

    def run_skill(self, ctx):
        path = Path(__file__).parents[1] / 'grutopia_extension/interaction_clearance/skills/composable_clear_blockage.py'
        skill, _ = load_skill(path)
        return skill(ctx)

    def test_planning_failure_calls_separate_base_policy_before_retry(self):
        ctx, calls = self.scenario(failures=1)
        self.assertTrue(self.run_skill(ctx).success)
        self.assertEqual([n for n, _ in calls[:5]], ['base_move', 'arm_plan', 'base_move', 'arm_plan', 'grasp_lift'])
        self.assertEqual(calls[2][1], {'target':'grasp', 'route':'auto', 'grasp_distance_offset_m':-.01})

    def test_two_retries_and_return_home_fit_original_twelve_call_budget(self):
        ctx, calls = self.scenario(failures=2, return_home=True)
        self.assertTrue(self.run_skill(ctx).success)
        self.assertEqual(len(calls), 11)
        self.assertEqual([o['grasp_distance_offset_m'] for _, o in calls if 'grasp_distance_offset_m' in o], [-.01, -.02])

    def test_exhausted_planning_stops_without_touching_object(self):
        ctx, calls = self.scenario(failures=3)
        self.assertEqual(self.run_skill(ctx).reason, 'ARM_PLAN_REJECTED')
        self.assertEqual(len(calls), 6)
        self.assertNotIn('grasp_lift', [n for n, _ in calls])

    def test_blocked_base_adjustment_stops_without_another_arm_attempt(self):
        ctx, calls = self.scenario(failures=1, blocked=True)
        self.assertEqual(self.run_skill(ctx).reason, 'BASE_PLAN_REJECTED')
        self.assertEqual(len(calls), 3)

    def test_physical_lift_failure_is_never_retried(self):
        ctx, calls = self.scenario(lift_failed=True)
        self.assertEqual(self.run_skill(ctx).reason, 'ARM_EXECUTION_FAILURE')
        self.assertEqual(len(calls), 3)

    def test_other_object_policies_keep_original_failure_behavior(self):
        for kind in ('box', 'bin'):
            ctx, calls = self.scenario(failures=3, kind=kind)
            self.assertFalse(self.run_skill(ctx).success)
            self.assertEqual(len(calls), 2)

    def test_offset_cannot_inject_unbounded_or_wrong_policy_goals(self):
        ctx, calls = self.scenario()
        for value in (float('nan'), float('inf'), -.021, .001, '0'):
            self.assertEqual(ctx.execute('base_move', target='grasp', grasp_distance_offset_m=value).reason, 'INVALID_POLICY_ARGUMENTS')
        for name, options in [('base_move', {'target':'placement'}), ('arm_plan', {'mode':'grasp'}), ('grasp_lift', {})]:
            self.assertEqual(ctx.execute(name, grasp_distance_offset_m=-.01, **options).reason, 'INVALID_POLICY_ARGUMENTS')
        self.assertEqual(calls, [])


class BaseOffsetTests(unittest.TestCase):
    def environment(self):
        yaw = .7
        q = Rotation.from_euler('z', yaw).as_quat()[[3,0,1,2]]
        return SimpleNamespace(task=ClearanceTask(object_kind='stool'), interaction_state='ready', arm_plan=None,
            object=SimpleNamespace(get_world_pose=lambda:(np.array((.8,.2,.0001)), q)),
            base=SimpleNamespace(get_world_pose=lambda:(np.zeros(3), np.array((1.,0.,0.,0.)))))

    def test_target_correction_follows_object_heading_and_preserves_task(self):
        env = self.environment()
        before = env.task.to_dict()
        policy = BaseMovePolicy(env)
        nominal = policy.target('grasp')
        adjusted = policy.target('grasp', grasp_distance_offset_m=-.01)
        np.testing.assert_allclose(adjusted[:2]-nominal[:2], .01*np.array((np.cos(.7),np.sin(.7))), atol=1e-12)
        self.assertEqual(env.task.to_dict(), before)
        env.interaction_state = 'holding'
        with self.assertRaises(ValueError): policy.target('grasp', grasp_distance_offset_m=-.01)

    def test_collision_rejection_prevents_any_adjustment_motion(self):
        env = self.environment()
        policy = BaseMovePolicy(env)
        with patch.object(policy, 'plan', side_effect=RuntimeError('unsafe path')), patch.object(policy, 'drive_segment') as drive:
            result = policy.execute(env.task, target='grasp', grasp_distance_offset_m=-.01)
        self.assertEqual(result.reason, 'BASE_PLAN_REJECTED')
        drive.assert_not_called()


class RecordedBaseReachabilityTests(unittest.TestCase):
    def environment(self, data, directory):
        urdf = robot_urdf()
        model, kin = RobotCollisionModel(urdf), BimanualUrdfKinematics(urdf)
        task = ClearanceTask.from_dict(data['task'])
        o = data['observation']
        pose = np.asarray(o['base_pose_world']).copy()
        joints = np.asarray(o['joint_positions']).copy()
        env = SimpleNamespace(task=task, collision_model=model, bimanual=kin, interaction_state='ready', arm_plan=None,
                recorder=SimpleNamespace(event=Mock()), output=Path(directory), elapsed_steps=1351,
                box=BimanualBox(task.dimensions, task.mass_kg, grasp_height(task.object_kind, task.dimensions), .003, task.object_kind),
                initial_control=np.asarray(data['initial_observation']['joint_positions']), control_indices=np.arange(19),
                robot=SimpleNamespace(dof_names=kin.names+[f'{s}_gripper_finger_joint{i}' for s in ('left','right') for i in (1,2)],
                                      get_joint_positions=lambda:joints.copy()),
                base=SimpleNamespace(get_world_pose=lambda:(pose[:3].copy(),pose[3:].copy())),
                object=SimpleNamespace(get_world_pose=lambda:(np.array(o['object_pose_world'][:3]),np.array(o['object_pose_world'][3:]))),
                rotation=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix())
        env.wall_objects = {w.name:model.fcl.CollisionObject(model.fcl.Box(*w.dimensions),
                model.fcl.Transform(yaw_rotation(w.yaw_rad),w.position)) for w in task.scene_obstacles}
        return env, pose, joints

    def test_original_solver_has_safe_base_and_arm_paths_after_adjustment(self):
        data = json.loads((Path(__file__).parent / 'fixtures/r28_pregrasp_failure.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            env, pose, joints = self.environment(data, directory)
            task = env.task
            registry = primitive_registry(env)
            destination = registry['base_move'].target('grasp', grasp_distance_offset_m=-.01)
            path = registry['base_move'].plan(destination, 'auto', False)
            np.testing.assert_allclose(path[-1], destination)
            # Numerical replay of the endpoint and a nearby tracking residual,
            # not a physical motion or a changed initial task configuration.
            for residual in (0., .003):
                with self.subTest(residual=residual):
                    pose[:2] = destination[:2]-residual*yaw_rotation(destination[2])[:2,0]
                    result = registry['arm_plan'].execute(task)
                    self.assertTrue(result.success, result.details)
                    audit = json.loads(next(Path(directory).glob('arm_plan_grasp_*.json')).read_text())
                    self.assertGreaterEqual(min(r['minimum_noncontact_clearance_m'] for r in audit), .015)
                    self.assertGreaterEqual(min(r['minimum_wall_clearance_m'] for r in audit), .08)
                    self.assertTrue(all(not r['self_intersections'] for r in audit))
                    np.testing.assert_array_equal(env.robot.get_joint_positions(), joints)
            self.assertEqual(json.loads(json.dumps(task.to_dict())), data['task'])

    def test_original_later_approach_candidate_keeps_its_successful_posture(self):
        data = json.loads((Path(__file__).parent / 'fixtures/r28_left_wide_pregrasp.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            env, _, joints = self.environment(data, directory)
            result = primitive_registry(env)['arm_plan'].execute(env.task)
            self.assertTrue(result.success, result.details)
            self.assertEqual(env.arm_plan['approach_clearance_m'], .09)
            self.assertEqual([r['approach_clearance_m'] for r in env.arm_plan['rejected_approaches']], [.13, .11])
            np.testing.assert_array_equal(env.robot.get_joint_positions(), joints)


if __name__ == '__main__':
    unittest.main()
