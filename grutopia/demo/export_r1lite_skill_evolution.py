"""Download four successful, independently audited layout examples and export a mosaic."""

import argparse
import json
import math
from pathlib import Path
import shutil
import shlex
import subprocess
import tarfile

from PIL import Image, ImageDraw, ImageFont

from export_r1lite_bimanual_visualization import validate_video


def download_episode(host, remote, destination, reuse=False):
    if destination.exists():
        if reuse and all((destination / name).exists() for name in
                         ('result.json', 'offline_reaudit.json', 'source_manifest.json', 'demonstration.npz', 'clearance_layout.mp4')):
            return
        raise ValueError('Use a new artifact directory, not a partial or previous export')
    destination.mkdir(parents=True)
    archive = destination.parent / (destination.name + '.transfer.tar.gz')
    with archive.open('wb') as stream:
        subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                        f'tar -C {shlex.quote(remote)} -czf - .'], stdout=stream, check=True, timeout=300)
    with tarfile.open(archive, 'r:gz') as package:
        resolved = destination.resolve()
        for member in package.getmembers():
            if (not (resolved / member.name).resolve().is_relative_to(resolved)
                    or not (member.isfile() or member.isdir()) or member.issym() or member.islnk()):
                raise ValueError('Unsafe archive member: ' + member.name)
        package.extractall(destination)
    archive.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--host', default='zkj@162.105.195.166')
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
    parser.add_argument('--reuse-downloads', action='store_true')
    parser.add_argument('--replace-mosaic', action='store_true')
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('ffmpeg is required')
    root = args.experiment
    report = json.loads((root / 'evolution_report.json').read_text())
    champion = report['selected_candidate']
    history = json.loads((root / 'training_history.json').read_text())
    train = next(item['train'] for item in history['candidates'] if item['candidate_id'] == champion)
    validation = json.loads((root / f'{champion}_validation.json').read_text())
    heldout = json.loads((root / f'{champion}_heldout.json').read_text())
    pool = [('train', item) for item in train['results']] + [('validation', item) for item in validation['results']] + [('heldout', item) for item in heldout['results']]
    chosen = []
    names = ('pose_left_standard', 'pose_right_standard', 'pose_left_offset', 'pose_right_long') if report.get('initial_pose_variations') else ('left_standard', 'right_standard', 'left_offset', 'right_long')
    for name in names:
        match = next(((split, case) for split, case in pool if case['layout_id'] == name and case['success']), None)
        if match:
            chosen.append(match)
    for split, case in pool:
        if case['success'] and all(case['layout_id'] != existing['layout_id'] for _, existing in chosen):
            chosen.append((split, case))
        if len(chosen) >= 4:
            break
    chosen = chosen[:4]
    if len(chosen) < 4:
        raise RuntimeError('Four distinct successful audited layouts are required for the success mosaic')
    media = root / 'videos'
    media.mkdir(exist_ok=True)
    sources, exports, durations, validations, pose_labels = [], [], [], [], []
    comparison = Image.new('RGB', (1920, 780 * len(chosen)), (245, 245, 245))
    font = ImageFont.truetype(args.font, 26)
    pose_font = ImageFont.truetype(args.font, 16)
    drawing = ImageDraw.Draw(comparison)
    examples = []
    for index, (split, case) in enumerate(chosen):
        run = root / 'sim_data' / case['layout_id']
        download_episode(args.host, case['run'], run, args.reuse_downloads)
        result = json.loads((run / 'result.json').read_text())
        audit = json.loads((run / 'offline_reaudit.json').read_text())
        if not result['success'] or not audit['passed']:
            raise RuntimeError('Downloaded episode failed verification')
        source = run / 'clearance_layout.mp4'
        validate_video(source, result['frames'], args.ffmpeg)
        export = media / (case['layout_id'] + '.mp4')
        if not export.exists():
            subprocess.run([args.ffmpeg, '-v', 'error', '-nostdin', '-n', '-i', str(source), '-an', '-c:v', 'libx264',
                            '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(export)], check=True)
        validations.append(validate_video(export, result['frames'], args.ffmpeg))
        sources.append(export)
        exports.append(case['layout_id'])
        durations.append(result['frames'] / 50)
        pose_label = ''
        if report.get('initial_pose_variations'):
            task = result['task']
            pose_label = (f"R XY=({task['robot_xy'][0]:+.2f},{task['robot_xy'][1]:+.2f}) "
                          f"yaw={math.degrees(task['robot_yaw_rad']):+.0f}deg | "
                          f"Box XY=({task['object_xy'][0]:+.2f},{task['object_xy'][1]:+.2f}) "
                          f"yaw={math.degrees(task['object_yaw_rad']):+.0f}deg")
        pose_labels.append(pose_label)
        top = 780 * index
        comparison.paste(Image.open(run / 'start_layout.jpg').convert('RGB'), (0, top + 60))
        finished = run / 'finish_layout.jpg'
        comparison.paste(Image.open(finished if finished.exists() else run / 'settle_layout.jpg').convert('RGB'), (960, top + 60))
        drawing.text((24, top + (4 if pose_label else 14)), f'{case["layout_id"]} [{split}]：清障前', font=font, fill=(20, 20, 20))
        if pose_label:
            drawing.text((24, top + 40), pose_label, font=pose_font, fill=(20, 20, 20))
        drawing.text((984, top + 14), '清障后：物体放稳、松手，指定通行区已清出', font=font, fill=(20, 20, 20))
        examples.append({'layout_id': case['layout_id'], 'split': split, 'remote_run': case['run'], 'local_run': str(run)})
    comparison.save(root / '02_多布局清障前后.jpg', quality=95)
    maximum = max(durations)
    expected_frames = round(maximum * 50)
    filters = []
    filter_font = args.font.replace('\\', '/').replace(':', '\\:')
    for index, (duration, name) in enumerate(zip(durations, exports)):
        pose_overlay = (f",drawtext=fontfile='{filter_font}':text='{pose_labels[index]}':x=12:y=452:"
                        'fontsize=13:fontcolor=white:box=1:boxcolor=black@0.6') if pose_labels[index] else ''
        filters.append(f'[{index}:v]scale=640:480,setsar=1,tpad=stop_mode=clone:stop_duration={maximum-duration+1:.3f},'
                       f"trim=end_frame={expected_frames},settb=1/50,setpts=N,drawtext=fontfile='{filter_font}':text={name}:x=12:y={424 if pose_labels[index] else 440}:"
                       f'fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6{pose_overlay}[v{index}]')
    filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|640_0|0_480|640_480[out]')
    mosaic = root / '01_多布局清障概览.mp4'
    command = [args.ffmpeg, '-v', 'error', '-nostdin', '-y' if args.replace_mosaic else '-n']
    for source in sources:
        command.extend(['-i', str(source)])
    command.extend(['-filter_complex_threads', '1', '-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                    '-r', '50', '-fps_mode', 'cfr', '-frames:v', str(expected_frames),
                    '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mosaic)])
    subprocess.run(command, check=True)
    validations.append(validate_video(mosaic, expected_frames, args.ffmpeg))
    output = {'passed': True, 'examples': examples, 'videos': validations,
              'mosaic_shorter_episodes_padded_with_final_frame': True, 'all_individual_videos_continuous': True}
    (root / 'video_validation.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'passed': True, 'videos': len(validations), 'layouts': exports}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
