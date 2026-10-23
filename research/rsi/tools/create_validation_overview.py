import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from research.rsi.tools.create_campaign_montage import create_page, FFMPEG, FAMILIES, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='round-06')
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.round.replace('-', '').isalnum():
        raise ValueError('Invalid round name')
    batch = args.results_root.resolve() / args.round / 'validation'
    output = batch / '多窗口汇总'
    report = json.loads((batch / 'round_report.json').read_text(encoding='utf-8'))
    if report['completed'] != 60 or report['total'] != 60:
        raise ValueError('Complete 60-case validation report required')
    cases = sorted(report['results'], key=lambda row: (list(FAMILIES).index(row['family']), row['repeat']))
    if len({row['case_id'] for row in cases}) != 60:
        raise ValueError('Validation case coverage mismatch')
    pages = []
    for page in range(1, 6):
        video = output / f'page_{page:02d}_12窗口_最终结果_2倍速.mp4'
        metadata = json.loads(video.with_suffix('.validation.json').read_text(encoding='utf-8'))
        if not metadata['artifact_integrity_passed'] or hashlib.sha256(video.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('Montage page integrity mismatch')
        if metadata['runtime_sha256'] != report['runtime_sha256'] or metadata['skill_sha256'] != report['skill_sha256']:
            raise ValueError('Montage page source version mismatch')
        pages.append(metadata)
    shown = [case['case_id'] for page in pages for case in page['source_cases']]
    if shown != [case['case_id'] for case in cases]:
        raise ValueError('Montage omitted or duplicated a validation case')
    if sum(case['success'] for page in pages for case in page['source_cases']) != report['successes']:
        raise ValueError('Montage outcome count mismatch')
    for page in pages:
        for case in page['source_cases']:
            if hashlib.sha256(Path(case['source']).read_bytes()).hexdigest() != case['source_sha256']:
                raise ValueError('Original video/screenshot changed')
    playlist = output / '全部60例.ffconcat'
    playlist.write_text('ffconcat version 1.0\n' + ''.join(f"file '{page['file']}'\n" for page in pages), encoding='utf-8')
    complete = output / '独立验证集_全部60例_12窗口分页_成功56失败4_2倍速.mp4'
    subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(playlist), '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart', str(complete)], check=True)
    probe = subprocess.run([str(FFMPEG.with_name('ffprobe.exe')), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,width,height,nb_read_frames:format=duration', '-of', 'json', str(complete)],
                           check=True, capture_output=True, text=True)
    metadata = json.loads(probe.stdout)
    expected_frames = sum(page['decoded_frames'] for page in pages)
    stream = metadata['streams'][0]
    if int(stream['nb_read_frames']) != expected_frames or stream['codec_name'] != 'h264':
        raise ValueError('Combined video frame count mismatch')
    if [stream['width'], stream['height']] != pages[0]['resolution']:
        raise ValueError('Combined video dimensions mismatch')
    chapters, timestamp = [], 0.0
    for page in pages:
        chapters.append({'page': page['page'], 'start_seconds': timestamp, 'cases': [case['case_id'] for case in page['source_cases']]})
        timestamp += page['encoded_seconds']
    document = {'file': complete.name, 'sha256': hashlib.sha256(complete.read_bytes()).hexdigest(),
                'artifact_integrity_passed': True, 'cases': 60, 'successes': report['successes'], 'failures': 60 - report['successes'],
                'rollout_videos': sum(case['presentation'] == 'recorded_rollout' for page in pages for case in page['source_cases']),
                'actual_no_action_scene_stills': sum(case['presentation'] == 'actual_scene_still_no_actions' for page in pages for case in page['source_cases']),
                'decoded_frames': expected_frames, 'encoded_seconds': float(metadata['format']['duration']),
                'resolution': [stream['width'], stream['height']], 'chapters': chapters,
                'runtime_sha256': report['runtime_sha256'], 'skill_sha256': report['skill_sha256'],
                'labels_are_final_not_live': True, 'simulation_results_unchanged': True, 'all_tasks_passed': False}
    complete.with_suffix('.validation.json').write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    failures = [case for case in cases if not case['success']]
    create_page(batch, report, failures, 6)
    print(json.dumps(document, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
