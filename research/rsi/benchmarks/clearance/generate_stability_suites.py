import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import FrozenBaseCollisionChecker
from grutopia_extension.interaction_clearance.task import ClearanceTask
from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox, BimanualUrdfKinematics


def quaternion(yaw):
    return Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-suite', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_suite.read_text())
    with (args.reference / 'transitions.jsonl').open() as stream:
        initial = np.asarray(json.loads(next(stream))['observation']['joint_positions'])
    model, kinematics = RobotCollisionModel(args.urdf), BimanualUrdfKinematics(args.urdf)
    names = kinematics.names + [f'{side}_gripper_finger_joint{index}' for side in ('left', 'right') for index in (1, 2)]
    args.output.mkdir(parents=True, exist_ok=True)
    for split, count, offset in [('development', 5, 340000), ('validation', 10, 8340000)]:
        destination = args.output / f'{split}_suite.json'
        if destination.exists():
            raise ValueError(f'Will not overwrite a previously frozen suite: {destination}')
        cases, rejected = [], []
        for family_index, family in enumerate(baseline['families']):
            template = ClearanceTask.from_dict(next(case['task'] for case in baseline['cases'] if case['family'] == family['name']))
            for repetition in range(count):
                base_seed = offset + family_index * 1000 + repetition
                for attempt in range(100):
                    seed = base_seed + attempt * 100000000
                    rng = np.random.default_rng(seed)
                    robot_xy = np.asarray(family['robot'][:2]) + rng.uniform(-0.08, 0.08, 2)
                    object_xy = np.asarray(family['box'][:2]) + rng.uniform(-0.035, 0.035, 2)
                    robot_yaw = np.radians(family['robot'][2] + rng.uniform(-15, 15))
                    object_yaw = np.radians(family['box'][2] + rng.uniform(-8, 8))
                    task = replace(template, task_id=f"{family['name']}_{split}_{repetition:02d}", seed=int(seed),
                                   robot_xy=tuple(robot_xy), object_xy=tuple(object_xy), robot_yaw_rad=float(robot_yaw),
                                   object_yaw_rad=float(object_yaw), return_to_start=False)
                    try:
                        task.validate()
                        environment = SimpleNamespace(task=task, collision_model=model, box=BimanualBox(task.dimensions, task.mass_kg),
                            robot=SimpleNamespace(dof_names=names, get_joint_positions=lambda: initial),
                            base=SimpleNamespace(get_world_pose=lambda: (np.r_[robot_xy, 0.0], quaternion(robot_yaw))),
                            object=SimpleNamespace(get_world_pose=lambda: (np.r_[object_xy, 0.0001], quaternion(object_yaw))),
                            rotation=lambda value: Rotation.from_quat(np.asarray(value)[[1, 2, 3, 0]]).as_matrix())
                        environment.wall_objects = {item.name: model.fcl.CollisionObject(model.fcl.Box(*item.dimensions),
                            model.fcl.Transform(np.eye(3), item.position)) for item in task.scene_obstacles}
                        clearance = FrozenBaseCollisionChecker(environment, False, minimum=0.02).details(np.r_[robot_xy, robot_yaw])
                        if not clearance['accepted']:
                            raise ValueError('Initial geometric clearance below 20 mm')
                    except (ValueError, RuntimeError) as error:
                        rejected.append({'family': family['name'], 'repeat': repetition, 'seed': seed, 'reason': str(error)})
                        continue
                    cases.append({'case_id': task.task_id, 'family': family['name'], 'repeat': repetition,
                                  'sample_seed': seed, 'task': task.to_dict(), 'initial_clearance': clearance})
                    break
                else:
                    raise RuntimeError(f'Unable to sample legal initial state for {family["name"]}')
        suite = {'purpose': f'stability_{split}', 'cases': cases, 'families': baseline['families'],
                 'sampling_rejections': rejected, 'only_invalid_spawn_states_filtered': True,
                 'path_and_arm_feasibility_not_prefiltered': True, 'split': split,
                 'perturbations': {'robot_xy_each_axis_m': 0.08, 'box_xy_each_axis_m': 0.035,
                                   'robot_yaw_degrees': 15, 'box_yaw_degrees': 8},
                 'validation_used_for_optimization': False}
        destination.write_text(json.dumps(suite, indent=2))
        print(json.dumps({'split': split, 'cases': len(cases), 'invalid_spawn_rejections': len(rejected)}), flush=True)


if __name__ == '__main__':
    main()
