"""Fixed, disjoint layout suites and safety-first macro-averaged selection."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from grutopia_extension.interaction_clearance.task import ClearanceTask, Region, SceneObstacle
from dataclasses import replace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def runtime_fingerprint(root):
    names = ['task', 'environment', 'skill_runtime', 'recording', 'policies', 'primitives', 'motion_geometry', 'base_planner', 'placement']
    paths = [f'grutopia_extension/interaction_clearance/{name}.py' for name in names]
    paths += [f'grutopia_extension/manipulation/{name}.py' for name in
              ('bimanual_collision', 'bimanual_floor_lift', 'floor_lift', 'r1lite_lift_sim', 'r1lite_bimanual_lift_sim', 'local_ground')]
    paths += ['grutopia_extension/interactive_navigation/r1lite_kinematics.py',
              'grutopia_extension/configs/robots/r1lite.py', 'grutopia/demo/r1lite_interaction_clearance.py',
              'grutopia/demo/audit_r1lite_clearance.py', 'grutopia/demo/r1lite_clearance_multilayout.py',
              'grutopia_extension/interaction_clearance/benchmark.py']
    files = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in paths}
    return {'sha256': digest(files), 'files': files}


def default_suite():
    layouts = []
    configurations = (
        ('left_standard', 'train', 1.90, 0.55, -0.55, 1.55, 0.70, 1.50),
        ('right_standard', 'train', 1.90, 0.55, -0.55, 1.55, -0.70, 1.35),
        ('left_wide', 'train', 1.95, 0.66, -0.66, 1.35, 0.70, 1.25),
        ('left_offset', 'validation', 1.98, 0.60, -0.54, 1.50, 0.73, 1.40),
        ('right_wide', 'validation', 1.92, 0.63, -0.63, 1.40, -0.68, 1.30),
        ('left_long', 'heldout', 2.04, 0.59, -0.59, 1.55, 0.78, 1.45),
        ('right_long', 'heldout', 2.04, 0.57, -0.57, 1.50, -0.78, 1.20),
    )
    for index, (name, split, wall_x, north_y, south_y, length, place_y, mass) in enumerate(configurations):
        data = ClearanceTask().to_dict()
        data['task_id'] = name
        data['mass_kg'] = mass
        data['object_xy'] = [0.65 + (index % 3 - 1) * 0.002, (index % 2) * 0.002]
        data['place_xy'] = [data['object_xy'][0], place_y]
        data['seed'] = 100 + index
        data['placement'] = ({'lower': [0.32, 0.40], 'upper': [0.99, 1.12]} if place_y > 0
                             else {'lower': [0.32, -1.12], 'upper': [0.99, -0.40]})
        data['scene_obstacles'] = [
            {'name': 'NorthWall', 'position': [wall_x, north_y, 0.40], 'dimensions': [length, 0.10, 0.80]},
            {'name': 'SouthWall', 'position': [wall_x, south_y, 0.40], 'dimensions': [length, 0.10, 0.80]},
        ]
        if name == 'left_offset':
            data['passage'] = {'lower': [0.45, -0.31], 'upper': [1.8, 0.37]}
        layouts.append({'layout_id': name, 'split': split, 'task': data})
    return {'format_version': 1, 'suite_id': 'r1lite_multilayout_v1', 'repeats': 1, 'layouts': layouts,
            'scope': 'small_box_clearance_with_widened_manipulation_areas',
            'statistical_reliability_established': False}


def planner_regression_tasks():
    original = ClearanceTask.from_dict(pose_suite()['layouts'][0]['task'])
    baseline = replace(original, task_id='astar_pose_regression')
    pillar = SceneObstacle('CarryPillar', (0.70, 0.95, 0.40), (0.22, 0.22, 0.80))
    detour = replace(original, task_id='astar_carry_pillar', place_xy=(0.70, 1.85), seed=700,
                     placement=Region((0.30, 1.45), (1.10, 2.30)),
                     scene_obstacles=original.scene_obstacles + (pillar,))
    cabinet = SceneObstacle('SideCabinet', (-0.65, 0.82, 0.40), (0.25, 0.30, 0.80))
    clutter = replace(detour, task_id='astar_staggered_clutter', seed=701,
                      scene_obstacles=detour.scene_obstacles + (cabinet,))
    for task in (baseline, detour, clutter):
        task.validate()
    return (baseline, detour, clutter)


def validate_suite(suite):
    if suite.get('format_version') != 1 or not isinstance(suite.get('repeats'), int) or not 1 <= suite['repeats'] <= 10:
        raise ValueError('Suite requires format_version=1 and 1..10 paired repeats')
    layouts = suite.get('layouts', [])
    if not 3 <= len(layouts) <= 30:
        raise ValueError('Suite requires 3..30 layouts')
    names = set()
    geometry_splits = {}
    splits = set()
    for layout in layouts:
        name, split = layout['layout_id'], layout['split']
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError('Layout IDs must be unique identifiers')
        if split not in ('train', 'validation', 'heldout'):
            raise ValueError('Invalid layout split')
        task = ClearanceTask.from_dict(layout['task'])
        geometry = {'obstacles': task.to_dict()['scene_obstacles'], 'passage': task.to_dict()['passage'],
                    'placement': task.to_dict()['placement']}
        identity = digest(geometry)
        if identity in geometry_splits and geometry_splits[identity] != split:
            raise ValueError('The same layout geometry cannot leak across splits')
        geometry_splits[identity] = split
        names.add(name)
        splits.add(split)
    if splits != {'train', 'validation', 'heldout'}:
        raise ValueError('All three disjoint splits must be present')
    return suite


def pose_suite():
    suite = default_suite()
    suite['suite_id'] = 'r1lite_pose_primitives_v2'
    suite['policy_interface'] = 'composable_v1'
    poses = (
        ((-0.14, -0.08), -20, (0.70, 0.06), 12, 0.78),
        ((-0.18, 0.10), 15, (0.72, -0.04), -12, -0.78),
        ((-0.20, -0.12), 28, (0.74, 0.03), 0, 0.80),
        ((-0.12, 0.15), -12, (0.71, -0.02), -16, 0.80),
        ((-0.22, -0.10), 22, (0.73, 0.05), 16, -0.80),
        ((-0.16, -0.14), -25, (0.76, 0.07), 18, 0.82),
        ((-0.24, 0.12), 30, (0.69, -0.06), -18, -0.82),
    )
    for index, (layout, (robot_xy, robot_degrees, object_xy, object_degrees, place_y)) in enumerate(zip(suite['layouts'], poses)):
        task = layout['task']
        task['robot_xy'] = list(robot_xy)
        task['robot_yaw_rad'] = math.radians(robot_degrees)
        task['object_xy'] = list(object_xy)
        task['object_yaw_rad'] = math.radians(object_degrees)
        task['place_xy'] = [object_xy[0], place_y]
        task['placement'] = ({'lower': [0.30, 0.40], 'upper': [1.08, 1.25]} if place_y > 0
                             else {'lower': [0.30, -1.25], 'upper': [1.08, -0.40]})
        task['seed'] = 300 + index
        for obstacle in task['scene_obstacles']:
            obstacle['position'][0] += 0.10
        layout['layout_id'] = 'pose_' + layout['layout_id']
        task['task_id'] = layout['layout_id']
    suite['scope'] = 'translated_and_rotated_robot_and_box_with_composable_physical_policies'
    return validate_suite(suite)


def suite_cases(suite, split):
    validate_suite(suite)
    if split not in ('train', 'validation', 'heldout'):
        raise ValueError('Invalid split')
    cases = []
    for layout in suite['layouts']:
        if layout['split'] != split:
            continue
        for repeat in range(suite['repeats']):
            task = dict(layout['task'])
            task['seed'] = int(task.get('seed', 0)) + repeat * 1000
            cases.append({'case_id': f'{layout["layout_id"]}_{repeat:02d}', 'layout_id': layout['layout_id'],
                          'repeat': repeat, 'task': task})
    return cases


def aggregate_results(cases, results):
    expected = {case['case_id']: case for case in cases}
    received = {result['case_id']: result for result in results}
    if len(received) != len(results) or set(received) != set(expected):
        raise ValueError('Selection requires exactly one result for every scheduled case')
    groups = {}
    for identifier, result in received.items():
        name = expected[identifier]['layout_id']
        groups.setdefault(name, []).append(result)
    per_layout = {name: {'episodes': len(items), 'successes': sum(bool(item['success']) for item in items),
                         'success_rate': sum(bool(item['success']) for item in items) / len(items)}
                  for name, items in groups.items()}
    successes = [item for item in results if item['success']]
    safety_failures = sum(bool(item.get('safety_failure', True)) for item in results)
    mean_time = sum(item['simulated_seconds'] for item in successes) / len(successes) if successes else 1e9
    mean_calls = sum(item.get('policy_calls', 0) for item in successes) / len(successes) if successes else 1e9
    macro = sum(item['success_rate'] for item in per_layout.values()) / len(per_layout)
    worst = min(item['success_rate'] for item in per_layout.values())
    score = [int(safety_failures == 0), -safety_failures / len(results), worst, macro, -mean_time, -mean_calls]
    return {'episodes': len(results), 'successes': len(successes), 'macro_success_rate': macro,
            'worst_layout_success_rate': worst, 'safety_failures': safety_failures,
            'mean_success_seconds': mean_time if successes else None, 'mean_success_policy_calls': mean_calls if successes else None,
            'per_layout': per_layout, 'failure_reasons': dict(Counter(item['reason'] for item in results if not item['success'])),
            'selection_score': score, 'statistical_reliability_established': False}


def select_best(evaluations):
    if not evaluations:
        raise ValueError('No candidates to select')
    identities = {(item['suite_sha256'], item['split'], item['runtime_sha256']) for item in evaluations}
    if len(identities) != 1:
        raise ValueError('Cannot compare different suites, splits, or runtime versions')
    return max(evaluations, key=lambda item: tuple(item['metrics']['selection_score']))
