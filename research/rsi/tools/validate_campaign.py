import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from research.rsi.paths import default_results_root


ROOT = default_results_root()
STAGES = {'targeted-regression': 2, 'full-regression': 12, 'development': 30, 'validation': 60, 'repeatability': 18}
FAMILY_COUNTS = {'full-regression': 2, 'development': 5, 'validation': 10, 'repeatability': 3}
FFPROBE = Path(r'D:\Zhou\ffmpeg-7.1-essentials_build\bin\ffprobe.exe')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sync_suite(round_name, stage, destination):
    remote = f'zkj@162.105.195.166:/data20t/embodied/zkj/inter-nav-results/r1lite-stability-v4/{round_name}/{stage}/suite.json'
    subprocess.run(['scp', '-q', remote, str(destination)], check=True)


def validate_case(batch, row, expected_task, manifest, links, skill_hash):
    case_id = row['case_id']
    run = batch / 'sim_data' / case_id
    result = read_json(run / 'result.json')
    audit = read_json(run / 'offline_reaudit.json')
    require(row['identity_matches'], f'Identity rejected: {case_id}')
    require(digest(result['task']) == digest(expected_task), f'Task mismatch: {case_id}')
    require(digest(read_json(run / 'task.json')) == digest(expected_task), f'Recorded task mismatch: {case_id}')
    require(result['skill_sha256'] == skill_hash == checksum(run / 'executed_skill.py'), f'Skill mismatch: {case_id}')
    snapshot = read_json(run / 'source_manifest.json')
    for relative, expected in manifest['files'].items():
        require(snapshot[relative] == expected == checksum(run / 'source' / relative), f'Source mismatch: {case_id}/{relative}')
    require(not result['object_attachment'] and not result['object_teleport_during_rollout'], f'Attachment/teleport claim: {case_id}')
    require(not result['training_performed'] and result['scope'] == 'mandatory_interaction_clearance_only', f'Unexpected scope: {case_id}')
    require(not result['navigation_decision_performed'], f'Unexpected bypass decision: {case_id}')
    require(audit['dynamic_object'] and not audit['gravity_disabled'] and not audit['object_attachment'], f'Object physics invalid: {case_id}')
    require(abs(audit['usd_mass_kg'] - expected_task['mass_kg']) < 1e-6, f'Mass mismatch: {case_id}')
    frames = result['frames']
    data = np.load(run / 'demonstration.npz')
    require(data['observations'].shape == (frames, 53) and data['actions'].shape == (frames, 22), f'Dataset shape: {case_id}')
    require(np.isfinite(data['observations']).all() and np.isfinite(data['actions']).all(), f'Nonfinite dataset: {case_id}')
    eligible = bool(result['success'] and result['collision_audit_passed'])
    require(bool(data['training_eligible']) == eligible, f'Eligibility mismatch: {case_id}')
    metadata = read_json(run / 'dataset_metadata.json')
    require(metadata['training_eligible'] == eligible and metadata['frames'] == frames, f'Metadata mismatch: {case_id}')
    require(abs(result['simulated_seconds'] - frames * 0.02) < 1e-6, f'Control clock mismatch: {case_id}')
    phases = metadata['phases']
    observed_phases = set()
    with (run / 'transitions.jsonl').open(encoding='utf-8') as stream:
        count = 0
        for count, line in enumerate(stream, 1):
            transition = json.loads(line)
            require(transition['step'] == count - 1, f'Transition sequence: {case_id}')
            before = transition['observation']
            onehot = np.zeros(len(phases))
            onehot[phases.index(transition['phase'])] = 1
            vector = np.r_[before['joint_positions'], before['object_pose_in_base'], before['placement_goal_in_base'],
                           before['measured_body_velocity'], np.asarray(before['hand_normal_force_vectors']).ravel(),
                           expected_task['dimensions'], onehot, transition['progress']].astype(np.float32)
            require(np.array_equal(vector, data['observations'][count - 1]), f'Observation mismatch: {case_id}/{count}')
            require(np.array_equal(np.asarray(transition['action'], dtype=np.float32), data['actions'][count - 1]), f'Action mismatch: {case_id}/{count}')
            observed_phases.add(transition['phase'])
    require(count == frames, f'Transition frame count: {case_id}')
    recording = read_json(run / 'physics_collision_samples.json')
    require(set(recording['link_names']) == links, f'Incomplete robot links: {case_id}')
    require(len(recording['samples']) == result['physics_samples'] == audit['physics_samples'] == frames * 4 + 40,
            f'Physics sample count: {case_id}')
    body_min, wall_min, finger_max, violations = float('inf'), float('inf'), 0.0, []
    for index, sample in enumerate(recording['samples']):
        require(sample['sample'] == index, f'Physics sequence gap: {case_id}')
        positions = np.asarray(sample['link_positions'])
        quaternions = np.asarray(sample['link_quaternions_wxyz'])
        require(positions.shape == (len(links), 3) and quaternions.shape == (len(links), 4), f'Physics link dimensions: {case_id}')
        require(np.isfinite(positions).all() and np.isfinite(quaternions).all(), f'Nonfinite physics: {case_id}')
        require(np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1)) < 1e-4, f'Invalid link quaternion: {case_id}')
        object_pose = np.r_[sample['object_position'], sample['object_quaternion_wxyz']]
        require(np.isfinite(object_pose).all(), f'Nonfinite object: {case_id}')
        require(abs(np.linalg.norm(object_pose[3:]) - 1) < 1e-4, f'Invalid object quaternion: {case_id}')
        require(np.isfinite([sample['minimum_noncontact_clearance_m'], sample['minimum_wall_clearance_m'],
                             sample['maximum_finger_penetration_m']]).all(), f'Nonfinite distance audit: {case_id}')
        body_min = min(body_min, sample['minimum_noncontact_clearance_m'])
        wall_min = min(wall_min, sample['minimum_wall_clearance_m'])
        finger_max = max(finger_max, sample['maximum_finger_penetration_m'])
        if sample['minimum_noncontact_clearance_m'] < 0.005 or sample['minimum_wall_clearance_m'] < 0.005 or sample['maximum_finger_penetration_m'] > 0.002 or sample['self_intersections']:
            violations.append(index)
    require(abs(body_min - audit['minimum_noncontact_clearance_m']) < 1e-5 and abs(wall_min - audit['minimum_wall_clearance_m']) < 1e-5,
            f'Audit extrema mismatch: {case_id}')
    if row['success']:
        require(result['success'] and audit['passed'] and not violations, f'False success: {case_id}')
        require({'lift', 'lower', 'release', 'retract'} <= observed_phases, f'Missing physical stages: {case_id}')
        metrics = audit['final_metrics']
        require(all(metrics[key] for key in ('success', 'road_clear', 'in_allowed_placement', 'object_supported', 'object_stationary', 'hands_released', 'safety_ok')),
                f'Clearance conditions incomplete: {case_id}')
        require(metrics['stable_seconds'] >= expected_task['required_stable_seconds'], f'Insufficient stability: {case_id}')
        require(audit['robot_passage_metrics']['robot_clear_of_passage'] and audit['released_state_ok'] and audit['base_stopped'], f'Robot/release conditions: {case_id}')
    video = batch / 'videos' / f'{case_id}.mp4'
    video_validation = read_json(video.with_suffix('.validation.json'))
    require(checksum(video) == video_validation['sha256'] and video_validation['episode_success'] == row['success'], f'Video hash/label: {case_id}')
    probe = subprocess.run([str(FFPROBE), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(video)],
                           check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    require(stream['codec_name'] == 'h264' and int(stream['nb_read_frames']) == frames, f'Video frames: {case_id}')
    return {'case_id': case_id, 'task_success': row['success'], 'frames': frames, 'physics_samples': len(recording['samples']),
            'body_min_m': body_min, 'wall_min_m': wall_min, 'finger_max_m': finger_max, 'violations': violations}


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_root.resolve()
    require(args.round.replace('-', '').isalnum(), 'Invalid round name')
    round_root = ROOT / args.round
    manifest = read_json(round_root / 'runtime_manifest.json')
    require(manifest['sha256'] == digest(manifest['files']), 'Corrupt runtime manifest')
    assets = read_json(round_root / 'robot_asset_manifest.json')
    require(all(checksum(round_root / 'robot_snapshot' / relative) == expected for relative, expected in assets.items()), 'Robot asset mismatch')
    urdf = ET.parse(round_root / 'robot_snapshot/r1lite_navigation.urdf').getroot()
    links = {link.get('name') for link in urdf.findall('link') if link.find('visual') is not None or link.find('collision') is not None}
    reports, incomplete, validated, stage_suites = {}, [], [], {}
    for stage, required_count in STAGES.items():
        batch = round_root / stage
        report_file = batch / 'round_report.json'
        if not report_file.exists():
            incomplete.append(stage)
            if not args.partial:
                continue
            report_file = batch / 'progress.json'
        if not report_file.exists():
            continue
        report = read_json(report_file)
        if args.partial and not any((batch / 'videos').glob('*.validation.json')):
            reports[stage] = {'evaluated': len(report['results']), 'successes': report['successes'],
                              'required': required_count, 'complete': False, 'no_verified_physical_exports': True}
            continue
        if not (batch / 'suite.json').exists():
            sync_suite(args.round, stage, batch / 'suite.json')
        suite = read_json(batch / 'suite.json')
        stage_suites[stage] = suite
        require(suite['runtime_sha256'] == manifest['sha256'], f'Runtime mismatch: {stage}')
        expected = {case['case_id']: case for case in suite['cases']}
        for row in report['results']:
            if not (batch / 'videos' / f'{row["case_id"]}.validation.json').exists() and args.partial:
                continue
            validated.append(dict(stage=stage, **validate_case(batch, row, expected[row['case_id']]['task'], manifest, links, suite['skill_sha256'])))
            print(json.dumps({'validated': row['case_id'], 'success': row['success'], 'stage': stage}), flush=True)
        complete = report.get('completed') == required_count and report['successes'] == required_count and len(report['results']) == required_count
        require(len(expected) == required_count, f'Wrong suite size: {stage}')
        if stage in FAMILY_COUNTS:
            family_counts = {family: sum(case['family'] == family for case in expected.values()) for family in {case['family'] for case in expected.values()}}
            require(len(family_counts) == 6 and set(family_counts.values()) == {FAMILY_COUNTS[stage]}, f'Family coverage mismatch: {stage}')
        reports[stage] = {'evaluated': len(report['results']), 'successes': report['successes'], 'required': required_count, 'complete': complete}
    require(args.partial or not incomplete, f'Missing required stages: {incomplete}')
    require(len({suite['skill_sha256'] for suite in stage_suites.values()}) <= 1, 'Skill changed between evaluation stages')
    if 'full-regression' in stage_suites:
        templates = {case['family']: case['task'] for case in stage_suites['full-regression']['cases']}
        fixed_fields = ('dimensions', 'mass_kg', 'place_xy', 'passage', 'placement', 'scene_obstacles', 'grasp_distance_m')
        for stage, suite in stage_suites.items():
            for case in suite['cases']:
                reference = templates[case['family']]
                require(all(digest(case['task'][key]) == digest(reference[key]) for key in fixed_fields), f'Scene/capability changed: {stage}/{case["case_id"]}')
                require(case['task']['required_clearance_m'] >= 0.03 and case['task']['required_stable_seconds'] >= 1.0,
                        f'Weakened task threshold: {case["case_id"]}')
    if 'development' in stage_suites and 'validation' in stage_suites:
        development = stage_suites['development']['cases']
        validation = stage_suites['validation']['cases']

        def initial_identity(case):
            task = case['task']
            return digest({key: task[key] for key in ('robot_xy', 'robot_yaw_rad', 'object_xy', 'object_yaw_rad')})

        require(not {case['task']['seed'] for case in development} & {case['task']['seed'] for case in validation}, 'Development/validation seed overlap')
        require(not {initial_identity(case) for case in development} & {initial_identity(case) for case in validation}, 'Development/validation initial-state overlap')
    acceptance = (not incomplete and len(reports) == len(STAGES) and all(row['complete'] for row in reports.values())
                  and len(validated) == sum(STAGES.values()) and all(row['task_success'] for row in validated))
    document = {'artifact_integrity_passed': True, 'acceptance_complete': acceptance, 'partial_validation': args.partial,
                'missing_stages': incomplete, 'stage_results': reports, 'validated_cases': validated,
                'runtime_sha256': manifest['sha256'], 'continuous_time_or_real_robot_guarantee': False}
    output = round_root / ('partial_artifact_validation.json' if args.partial else 'artifact_validation.json')
    output.write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(json.dumps({'cases_validated': len(validated), 'acceptance_complete': acceptance}), flush=True)


if __name__ == '__main__':
    main()
