"""Independently replay clearance geometry, placement, and dataset eligibility."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.task import ClearanceTask, evaluate_clearance
from grutopia_extension.interaction_clearance.placement import robot_passage_metrics
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--robot-urdf', default='/data20t/embodied/zkj/third_party/r1lite_isaac_v2/r1lite_navigation.urdf')
    args = parser.parse_args()
    from pxr import Usd, UsdGeom, UsdPhysics

    task = ClearanceTask.from_dict(json.loads((args.run / 'task.json').read_text()))
    result = json.loads((args.run / 'result.json').read_text())
    recording = json.loads((args.run / 'physics_collision_samples.json').read_text())
    model = RobotCollisionModel(args.robot_urdf)
    stage = Usd.Stage.Open(str(args.run / 'scene.usda'))
    wall_objects = {}
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/ClearanceScene/') and prim.HasAPI(UsdPhysics.CollisionAPI):
            if not prim.IsA(UsdGeom.Cube):
                raise ValueError('Auditor supports box-shaped scene obstacles only')
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            extent = np.linalg.norm(matrix[:3, :3], axis=0) * UsdGeom.Cube(prim).GetSizeAttr().Get()
            basis = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
            wall_objects[prim.GetName()] = model.fcl.CollisionObject(model.fcl.Box(*extent),
                                                                    model.fcl.Transform(basis, matrix[:3, 3]))
    if set(recording['link_names']) != set(model.names) or not wall_objects:
        raise ValueError('Incomplete link or scene-obstacle recording')
    violations = []
    minimum_body = float('inf')
    minimum_wall = float('inf')
    maximum_finger = 0.0
    request = model.fcl.DistanceRequest(enable_signed_distance=True)
    for sample in recording['samples']:
        transforms = {}
        for name, position, quaternion in zip(recording['link_names'], sample['link_positions'], sample['link_quaternions_wxyz']):
            transform = np.eye(4)
            transform[:3, :3] = rotation(quaternion)
            transform[:3, 3] = position
            transforms[name] = transform
        obstacle = np.eye(4)
        obstacle[:3, :3] = rotation(sample['object_quaternion_wxyz'])
        obstacle[:3, 3] = np.asarray(sample['object_position']) + obstacle[:3, :3] @ np.array((0, 0, task.dimensions[2] / 2))
        distances = model.box_clearances(transforms, obstacle, task.dimensions, include_fingers=True)
        body_gap = min(distances[name] for name in model.noncontact_names)
        finger_depth = max(0.0, -min(value for name, value in distances.items() if 'gripper_finger' in name))
        objects = model.objects(transforms)
        objects['task_object'] = model.fcl.CollisionObject(model.fcl.Box(*task.dimensions),
                                                         model.fcl.Transform(obstacle[:3, :3], obstacle[:3, 3]))
        wall_gap = min(float(model.fcl.distance(body, wall, request, model.fcl.DistanceResult()))
                       for body in objects.values() for wall in wall_objects.values())
        self_hits = model.self_intersections(transforms)
        minimum_body = min(minimum_body, body_gap)
        minimum_wall = min(minimum_wall, wall_gap)
        maximum_finger = max(maximum_finger, finger_depth)
        if body_gap < 0.005 or finger_depth > 0.002 or wall_gap < 0.005 or self_hits:
            violations.append({'sample': sample['sample'], 'body_gap_m': body_gap, 'wall_gap_m': wall_gap,
                               'finger_depth_m': finger_depth, 'self_intersections': self_hits})
    stable = 0.0
    frames = 0
    metrics = {}
    with (args.run / 'transitions.jsonl').open() as transitions:
        for line in transitions:
            transition = json.loads(line)
            observation = transition['next_observation']
            pose = observation['object_pose_world']
            metrics = evaluate_clearance(task, pose[:3], rotation(pose[3:]),
                                         float(np.linalg.norm(observation['object_linear_velocity'])),
                                         float(np.linalg.norm(observation['object_angular_velocity'])),
                                         float(np.linalg.norm(observation['hand_normal_force_vectors'], axis=1).sum()),
                                         not violations, stable)
            stable = stable + 0.02 if metrics['physically_complete'] else 0.0
            frames += 1
        metrics['stable_seconds'] = stable
        metrics['success'] = bool(metrics.get('physically_complete') and stable >= task.required_stable_seconds)
    object_prim = stage.GetPrimAtPath('/World/LiftObject')
    rigid = UsdPhysics.RigidBodyAPI(object_prim)
    dynamic = bool(rigid.GetRigidBodyEnabledAttr().Get()) and not rigid.GetKinematicEnabledAttr().Get()
    gravity_disabled = bool(object_prim.GetAttribute('physxRigidBody:disableGravity').Get())
    mass = float(UsdPhysics.MassAPI(object_prim).GetMassAttr().Get())
    attachment = any('/World/LiftObject' in str(target) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
                     for relationship in prim.GetRelationships() for target in relationship.GetTargets())
    data = np.load(args.run / 'demonstration.npz')
    metadata = json.loads((args.run / 'dataset_metadata.json').read_text())
    shapes_ok = data['observations'].shape == (frames, metadata['observation_size']) and data['actions'].shape == (frames, 22)
    finite_data = bool(np.isfinite(data['observations']).all() and np.isfinite(data['actions']).all())
    eligibility_ok = bool(data['training_eligible']) == bool(result['success'] and result['collision_audit_passed'])
    eligibility_ok = eligibility_ok and metadata['training_eligible'] == bool(data['training_eligible'])
    skill_matches = hashlib.sha256((args.run / 'executed_skill.py').read_bytes()).hexdigest() == metadata['skill_sha256']
    initialization_ok = bool(result.get('initialization', {}).get('passed', True))
    if any(abs(value) > 1e-9 for value in (*task.robot_xy, task.robot_yaw_rad, task.object_yaw_rad)):
        from grutopia_extension.interaction_clearance.motion_geometry import quaternion_yaw, angle_error
        first = recording['samples'][0]
        base_index = recording['link_names'].index('base_link')
        initialization_ok = (np.linalg.norm(np.asarray(first['link_positions'][base_index])[:2] - task.robot_xy) < 0.025
                             and np.linalg.norm(np.asarray(first['object_position'])[:2] - task.object_xy) < 0.025
                             and abs(angle_error(quaternion_yaw(first['link_quaternions_wxyz'][base_index]), task.robot_yaw_rad)) < 0.04
                             and abs(angle_error(quaternion_yaw(first['object_quaternion_wxyz']), task.object_yaw_rad)) < 0.04)
    protocol_ok = True
    if task.return_to_start:
        from grutopia_extension.interaction_clearance.motion_geometry import quaternion_yaw, angle_error
        base_pose = observation['base_pose_world']
        protocol_ok = (result.get('interaction_state') == 'released'
                       and np.linalg.norm(np.asarray(base_pose[:2]) - task.robot_xy) < 0.03
                       and abs(angle_error(quaternion_yaw(base_pose[3:]), task.robot_yaw_rad)) < 0.04)
    robot_metrics = robot_passage_metrics(task, model, transforms)
    released_ok = result.get('interaction_state') == 'released'
    base_stopped = np.linalg.norm(observation['measured_body_velocity'][:2]) < 0.02
    report = {'passed': bool(result['success'] and metrics['success'] and robot_metrics['robot_clear_of_passage']
                              and released_ok and base_stopped and not violations and dynamic and initialization_ok and protocol_ok
                              and not gravity_disabled and not attachment and abs(mass - task.mass_kg) < 1e-6
                              and shapes_ok and finite_data and eligibility_ok and skill_matches),
              'physics_samples': len(recording['samples']), 'frames': frames, 'violations': violations,
              'minimum_noncontact_clearance_m': minimum_body, 'minimum_wall_clearance_m': minimum_wall,
              'maximum_finger_penetration_m': maximum_finger, 'final_metrics': metrics,
              'dynamic_object': dynamic, 'gravity_disabled': gravity_disabled, 'object_attachment': attachment,
              'usd_mass_kg': mass, 'dataset_shapes_ok': shapes_ok, 'dataset_finite': finite_data,
              'dataset_eligibility_ok': eligibility_ok, 'executed_skill_hash_matches': skill_matches,
              'initial_poses_independently_checked': initialization_ok, 'base_return_protocol_ok': bool(protocol_ok),
              'robot_passage_metrics': robot_metrics, 'released_state_ok': released_ok, 'base_stopped': bool(base_stopped),
              'continuous_time_guarantee': False}
    (args.run / 'offline_reaudit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
