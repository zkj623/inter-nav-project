import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
from research.rsi.paths import default_results_root


ROOT = default_results_root()
REMOTE_ROOT = '/data20t/embodied/zkj/inter-nav-results/r1lite-stability-v4'
HOST = 'zkj@162.105.195.166'
FFMPEG = Path(r'D:\Zhou\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe')


def archive_no_action_case(local_case, row, videos):
    result = json.loads((local_case / 'result.json').read_text(encoding='utf-8'))
    if result['frames'] != 0:
        return False
    if row['success'] or result['success'] or not row['identity_matches'] or row['reason'] != 'BASE_PLAN_REJECTED':
        raise ValueError('Invalid zero-action failure identity')
    manifest = json.loads((local_case.parents[2] / 'runtime_manifest.json').read_text(encoding='utf-8'))
    source = json.loads((local_case / 'source_manifest.json').read_text(encoding='utf-8'))
    for relative, checksum in manifest['files'].items():
        if source.get(relative) != checksum or hashlib.sha256((local_case / 'source' / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Zero-action source mismatch: {relative}')
    data = np.load(local_case / 'demonstration.npz')
    if data['observations'].shape != (0, 53) or data['actions'].shape != (0, 22):
        raise ValueError('Zero-action dataset shape mismatch')
    if (local_case / 'transitions.jsonl').stat().st_size != 0:
        raise ValueError('Unexpected zero-action transitions')
    screenshot = local_case / 'finish_layout.jpg'
    metadata = {'case_id': row['case_id'], 'episode_success': False, 'reason': row['reason'],
                'source_identity_verified': True, 'artifact_integrity_passed': True,
                'offline_audit_passed': False, 'offline_audit_available': False,
                'control_frames': 0, 'video_exported': False, 'presentation': 'actual_scene_still_no_actions',
                'screenshot': screenshot.name, 'screenshot_sha256': hashlib.sha256(screenshot.read_bytes()).hexdigest()}
    (videos / f"{row['case_id']}.no_action.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return True


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_root.resolve()
    if not all(part.replace('-', '').isalnum() for part in args.batch.split('/')) or len(args.batch.split('/')) != 2:
        raise ValueError('Expected round-id/batch-name')
    remote = f'{REMOTE_ROOT}/{args.batch}'
    progress = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', HOST, f'cat {remote}/progress.json'],
                              check=True, capture_output=True).stdout
    rows = json.loads(progress)['results']
    output = ROOT / args.batch
    raw, videos = output / 'sim_data', output / 'videos'
    raw.mkdir(parents=True, exist_ok=True)
    videos.mkdir(exist_ok=True)
    (output / 'progress.json').write_bytes(progress)
    for row in rows:
        case_id = row['case_id']
        if not case_id.replace('_', '').isalnum():
            raise ValueError(f'Invalid case id: {case_id}')
        validation_file = videos / f'{case_id}.validation.json'
        unverified_file = videos / f'{case_id}.unverified.json'
        no_action_file = videos / f'{case_id}.no_action.json'
        if validation_file.exists() or unverified_file.exists() or no_action_file.exists():
            continue
        local_case = raw / case_id
        if row['identity_matches'] or subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', HOST, f'test -d {remote}/{case_id}']).returncode == 0:
            subprocess.run(['scp', '-q', '-r', f'{HOST}:{remote}/{case_id}', str(raw)], check=True)
        else:
            local_case.mkdir(exist_ok=True)
        subprocess.run(['scp', '-q', f'{HOST}:{remote}/{case_id}.log', str(local_case / 'simulation.log')], check=True)
        (output / f'{case_id}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        for attempt in row.get('startup_attempts', []):
            attempt_log = PurePosixPath(attempt['log'])
            prefix = f'{case_id}_attempt_'
            if str(attempt_log.parent) != remote or not attempt_log.stem.startswith(prefix) or not attempt_log.stem[len(prefix):].isdigit() or attempt_log.suffix != '.log':
                raise ValueError('Attempt log path is outside the evaluated batch')
            subprocess.run(['scp', '-q', f'{HOST}:{attempt_log}', str(local_case / attempt_log.name)], check=True)
        if not row['identity_matches']:
            unverified_file.write_text(json.dumps({'case_id': case_id, 'forensic_only': True,
                'episode_success': False, 'source_identity_verified': False, 'artifact_integrity_passed': False,
                'reason': row['reason'], 'video_exported': False, 'raw_logs_preserved': True}, indent=2), encoding='utf-8')
            print(f'Archived unverified/incomplete case without success claim: {case_id}', flush=True)
            continue
        result = json.loads((local_case / 'result.json').read_text(encoding='utf-8'))
        if archive_no_action_case(local_case, row, videos):
            print(f'Archived verified zero-action planning failure with real screenshot: {case_id}', flush=True)
            continue
        audit = json.loads((local_case / 'offline_reaudit.json').read_text(encoding='utf-8'))
        data = np.load(local_case / 'demonstration.npz')
        if data['observations'].shape != (result['frames'], 53) or data['actions'].shape != (result['frames'], 22):
            raise ValueError(f'Dataset shape mismatch: {case_id}')
        if not np.isfinite(data['observations']).all() or not np.isfinite(data['actions']).all():
            raise ValueError(f'Nonfinite dataset: {case_id}')
        if row['success'] and (not result['success'] or not audit['passed'] or audit['violations']):
            raise ValueError(f'Success label mismatch: {case_id}')
        video = videos / f'{case_id}.mp4'
        print(f'Exporting {args.batch}/{case_id}: success={row["success"]}', flush=True)
        subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(local_case / 'clearance_layout.mp4'), '-map', '0:v:0', '-an',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '4',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
        probe = subprocess.run([str(FFMPEG.with_name('ffprobe.exe')), '-v', 'error', '-count_frames',
                                '-select_streams', 'v:0', '-show_entries',
                                'stream=codec_name,width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(video)],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        if int(stream['nb_read_frames']) != result['frames'] or stream['codec_name'] != 'h264':
            raise ValueError(f'Video verification failed: {case_id}')
        validation_file.write_text(json.dumps({
            'file': video.name, 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
            'artifact_integrity_passed': True, 'decoded_frames': result['frames'], 'codec': stream['codec_name'],
            'fps': stream['r_frame_rate'], 'resolution': [stream['width'], stream['height']],
            'episode_success': row['success'], 'offline_audit_passed': audit['passed'],
            'source_identity_verified': row['identity_matches'], 'reason': row['reason'],
            'success_contract': 'object_released_and_stable_road_clear_robot_outside_passage',
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'batch': args.batch, 'completed_cases_exported': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
