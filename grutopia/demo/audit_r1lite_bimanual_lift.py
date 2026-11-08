"""Recompute a recorded bilateral lift's collision checks without running physics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.floor_lift import LiftSuccessTracker, lowest_box_point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--robot-urdf', required=True)
    args = parser.parse_args()
    from pxr import Usd, UsdPhysics

    result = json.loads((args.run / 'result.json').read_text())
    recording = json.loads((args.run / 'physics_collision_samples.json').read_text())
    runtime = json.loads((args.run / 'collision_audit.json').read_text())
    model = RobotCollisionModel(args.robot_urdf)
    names = recording['link_names']
    if set(names) != set(model.names) or not recording['samples']:
        raise ValueError('Incomplete geometry recording')
    minimum = float('inf')
    finger_depth = 0.0
    violations = []
    for sample in recording['samples']:
        transforms = {}
        for name, position, quaternion in zip(names, sample['link_positions'], sample['link_quaternions_wxyz']):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
            transform[:3, 3] = position
            transforms[name] = transform
        obstacle = np.eye(4)
        obstacle[:3, :3] = Rotation.from_quat(np.asarray(sample['object_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        obstacle[:3, 3] = np.asarray(sample['object_position']) + obstacle[:3, :3] @ np.array((
            0.0, 0.0, result['dimensions_m'][2] / 2
        ))
        distances = model.box_clearances(transforms, obstacle, result['dimensions_m'], include_fingers=True)
        gap = min(distances[name] for name in model.noncontact_names)
        depth = max(0.0, -min(value for name, value in distances.items() if 'gripper_finger' in name))
        minimum = min(minimum, gap)
        finger_depth = max(finger_depth, depth)
        self_hits = model.self_intersections(transforms)
        if gap < runtime['noncontact_clearance_threshold_m'] or depth > runtime['finger_penetration_tolerance_m'] or self_hits:
            violations.append({'sample': sample['sample'], 'gap_m': gap, 'finger_depth_m': depth,
                               'self_intersections': self_hits})
    stage = Usd.Stage.Open(str(args.run / 'scene.usda'))
    object_prim = stage.GetPrimAtPath('/World/LiftObject')
    mass = UsdPhysics.MassAPI(object_prim).GetMassAttr().Get()
    rigid = UsdPhysics.RigidBodyAPI(object_prim)
    attached = any('/World/LiftObject' in str(target)
                   for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
                   for relationship in prim.GetRelationships() for target in relationship.GetTargets())
    no_gravity = object_prim.GetAttribute('physxRigidBody:disableGravity').Get()
    dynamic = bool(rigid.GetRigidBodyEnabledAttr().Get()) and not rigid.GetKinematicEnabledAttr().Get()
    first_sample = recording['samples'][0]
    first_rotation = Rotation.from_quat(np.asarray(first_sample['object_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    first_center = np.asarray(first_sample['object_position']) + first_rotation @ np.array((
        0.0, 0.0, result['dimensions_m'][2] / 2
    ))
    tracker = LiftSuccessTracker(lowest_box_point(first_center, first_rotation, result['dimensions_m']))
    trajectory = json.loads((args.run / 'trajectory.json').read_text())
    for entry in trajectory:
        tracker.update(entry['bottom'], 0.02, entry['opposed_side_contact'], result['max_base_tilt_degrees'] < 12)
    report = {'passed': not violations and dynamic and not no_gravity and not attached
              and abs(float(mass) - result['mass_kg']) < 1e-6 and tracker.success and result['success']
              and trajectory[-1]['opposed_side_contact']
              and trajectory[-1]['bottom'] - tracker.initial_bottom >= tracker.required_lift,
              'recomputed_samples': len(recording['samples']), 'minimum_noncontact_clearance_m': minimum,
              'maximum_finger_penetration_m': finger_depth, 'violations': violations,
              'dynamic_object': dynamic, 'gravity_disabled': bool(no_gravity),
              'object_has_joint_attachment': attached, 'usd_mass_kg': mass,
              'lift_hold_recomputed': tracker.success, 'maximum_bottom_lift_m': tracker.max_lift,
              'hold_seconds': tracker.held_seconds, 'continuous_time_guarantee': False}
    (args.run / 'offline_reaudit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
