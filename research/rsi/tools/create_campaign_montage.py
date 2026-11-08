import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
from research.rsi.paths import default_results_root


ROOT = default_results_root()
FFMPEG = Path(r'D:\Zhou\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe')
FONT = r'C:\Windows\Fonts\msyh.ttc'
BOLD = r'C:\Windows\Fonts\msyhbd.ttc'
FAMILIES = {
    'left_wide_turn': '宽通道左侧放置',
    'right_reversed': '反向起步右侧放置',
    'staggered_shifted': '交错障碍与偏移放置',
    'offset_corridor_south': '非对称通道南侧放置',
    'narrow_entry_zigzag': '窄入口与较宽箱体',
    'long_offset_home': '远距离放置与目标障碍',
}
REASONS = {
    'ROAD_CLEARED': '放稳释放、清路且机器人不挡路',
    'ARM_PLAN_REJECTED': '双臂 IK 或轨迹约束拒绝',
    'BASE_PLAN_REJECTED': '底盘或放置配置规划拒绝',
    'ARM_EXECUTION_FAILURE': '抓取或放置执行失败',
    'BASE_EXECUTION_FAILURE': '底盘跟踪偏差或动作超时',
    'SAFETY_REJECTION': '触发安全间距停止',
}
STAGES = {'targeted-regression': '问题配置回归', 'full-regression': '12 场景回归',
          'development': '开发集', 'validation': '独立验证集', 'repeatability': '重启复测'}
FPS, SPEED = 25, 2
CELL_WIDTH, VIDEO_HEIGHT, CELL_HEADER, MAIN_HEADER = 640, 480, 100, 120


def text_fitted(drawing, position, label, size, color, width, bold=False):
    while True:
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        if drawing.textbbox((0, 0), label, font=font)[2] <= width or size <= 14:
            break
        size -= 1
    drawing.text(position, label, font=font, fill=color)


def run_logged(command, log):
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(log.read_text(encoding='utf-8', errors='replace')[-4000:])


def create_page(batch, report, selected, page):
    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    cell_height = CELL_HEADER + VIDEO_HEIGHT
    width, height = columns * CELL_WIDTH, MAIN_HEADER + rows * cell_height
    output = batch / '多窗口汇总'
    output.mkdir(exist_ok=True)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    drawing.rectangle((0, 0, width - 1, MAIN_HEADER - 1), fill='#111C2B')
    text_fitted(drawing, (20, 8), f'R1 Lite｜{STAGES.get(batch.name, batch.name)}｜同步回放', 32, 'white', width - 40, True)
    text_fitted(drawing, (20, 53), f"本评测 {report['successes']}/{report['total']} 通过｜第 {page} 页，展示 {len(selected)} 组｜2 倍速", 25, '#D6DFEC', width - 40)
    text_fitted(drawing, (20, 90), '绿色：任务及独立审计通过；红色：任务失败。标签为最终结果，非实时状态。', 21, '#D6DFEC', width - 40)
    cases = []
    for index, row in enumerate(selected):
        case_id = row['case_id']
        source = batch / 'videos' / f'{case_id}.mp4'
        still_metadata = batch / 'videos' / f'{case_id}.no_action.json'
        is_still = still_metadata.exists()
        if is_still:
            validation = json.loads(still_metadata.read_text(encoding='utf-8'))
            result = json.loads((batch / 'sim_data' / case_id / 'result.json').read_text(encoding='utf-8'))
            if row['success'] or result['frames'] != 0 or validation['control_frames'] != 0:
                raise ValueError(f'Invalid no-action scene still: {case_id}')
            source = batch / 'sim_data' / case_id / validation['screenshot']
            expected_hash = validation['screenshot_sha256']
        else:
            validation = json.loads(source.with_suffix('.validation.json').read_text(encoding='utf-8'))
            expected_hash = validation['sha256']
        if not validation['artifact_integrity_passed'] or not validation['source_identity_verified']:
            raise ValueError(f'Unverified source video: {case_id}')
        if validation['episode_success'] != row['success'] or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Video/result mismatch: {case_id}')
        result = json.loads((batch / 'sim_data' / case_id / 'result.json').read_text(encoding='utf-8'))
        if row['success'] and not row['offline_audit_passed']:
            raise ValueError(f'Success without independent audit: {case_id}')
        case = {'case_id': case_id, 'source': str(source), 'source_sha256': expected_hash,
                'success': row['success'], 'reason': row['reason'],
                'presentation': 'actual_scene_still_no_actions' if is_still else 'recorded_rollout',
                'end_seconds': 0 if is_still else validation['decoded_frames'] / float(Fraction(str(validation['fps']))) / SPEED}
        cases.append(case)
        left, top = (index % columns) * CELL_WIDTH, MAIN_HEADER + (index // columns) * cell_height
        color = '#58E39C' if row['success'] else '#FF7B7B'
        drawing.rectangle((left, top, left + CELL_WIDTH - 1, top + CELL_HEADER - 1), fill='#1C293A')
        drawing.rectangle((left, top, left + 4, top + cell_height - 1), fill=color)
        title = f"{index + 1:02d}｜{FAMILIES.get(row['family'], row['family'])} · 初态 {row['repeat'] + 1}"
        text_fitted(drawing, (left + 14, top + 3), title, 23, 'white', CELL_WIDTH - 28, True)
        status = '最终任务：成功｜独立审计通过' if row['success'] else '最终任务：失败｜' + REASONS.get(row['reason'], str(row['reason']))
        text_fitted(drawing, (left + 14, top + 34), status, 22, color, CELL_WIDTH - 28, True)
        gap = result.get('minimum_noncontact_clearance_m')
        description = f'{case_id}｜身物最小间距 {gap * 1000:.1f} mm' if gap is not None else case_id
        if is_still:
            description = f'{case_id}｜真实场景静帧：规划拒绝，无动作录像'
        text_fitted(drawing, (left + 14, top + 67), description, 17, '#D6DFEC', CELL_WIDTH - 28)
    label = output / f'page_{page:02d}_labels.png'
    overlay.save(label)
    ended = Image.new('RGBA', (250, 38), '#111C2B')
    ImageDraw.Draw(ended).text((10, 5), '已结束 · 保持最后一帧', font=ImageFont.truetype(FONT, 20), fill='white')
    ended_file = output / 'ended.png'
    ended.save(ended_file)
    duration = max(5.0, max(case['end_seconds'] for case in cases))
    video = output / f'page_{page:02d}_{len(cases)}窗口_最终结果_2倍速.mp4'
    command = [str(FFMPEG), '-hide_banner', '-y', '-filter_complex_threads', '4']
    for case in cases:
        if case['presentation'] == 'actual_scene_still_no_actions':
            command.extend(['-loop', '1', '-framerate', str(FPS)])
        command.extend(['-threads', '1', '-i', case['source']])
    for source in (label, ended_file):
        command.extend(['-loop', '1', '-framerate', str(FPS), '-i', str(source)])
    filters = []
    for index in range(len(cases)):
        filters.append(f'[{index}:v]setpts=(PTS-STARTPTS)/{SPEED},fps={FPS},scale={CELL_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,setsar=1,'
                       f'tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},pad={CELL_WIDTH}:{cell_height}:0:{CELL_HEADER}:color=0x1C293A[v{index}]')
    layout = '|'.join(f'{index % columns * CELL_WIDTH}_{index // columns * cell_height}' for index in range(len(cases)))
    filters.append(''.join(f'[v{index}]' for index in range(len(cases))) + f'xstack=inputs={len(cases)}:layout={layout}:fill=0x111C2B[grid]')
    filters.append(f'[grid]pad={width}:{height}:0:{MAIN_HEADER}:color=0x111C2B[canvas]')
    filters.append(f'[canvas][{len(cases)}:v]overlay=0:0:shortest=1[labeled]')
    marked_indices = [index for index, case in enumerate(cases) if case['presentation'] == 'recorded_rollout']
    if marked_indices:
        filters.append(f'[{len(cases) + 1}:v]split={len(marked_indices)}' + ''.join(f'[end{index}]' for index in marked_indices))
    previous = 'labeled'
    for index, case in enumerate(cases):
        if index not in marked_indices:
            continue
        left = index % columns * CELL_WIDTH + CELL_WIDTH - 264
        top = MAIN_HEADER + index // columns * cell_height + cell_height - 50
        current = f'marked{index}'
        filters.append(f"[{previous}][end{index}]overlay={left}:{top}:shortest=1:enable='gte(t,{case['end_seconds']:.6f})'[{current}]")
        previous = current
    graph = output / f'page_{page:02d}.fffilter'
    graph.write_text(';\n'.join(filters), encoding='utf-8')
    command.extend(['-filter_complex_script', str(graph), '-map', f'[{previous}]', '-an', '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-threads', '6', '-pix_fmt', 'yuv420p',
                    '-fps_mode', 'cfr', '-movflags', '+faststart', str(video)])
    print(f'Encoding montage page {page}: {len(cases)} cases, {duration:.2f}s', flush=True)
    run_logged(command, output / f'page_{page:02d}_encode.log')
    probe = subprocess.run([str(FFMPEG.with_name('ffprobe.exe')), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,width,height,nb_read_frames:format=duration', '-of', 'json', str(video)],
                           check=True, capture_output=True, text=True)
    metadata = json.loads(probe.stdout)
    stream = metadata['streams'][0]
    if (int(stream['nb_read_frames']) != math.ceil(duration * FPS) or (stream['width'], stream['height']) != (width, height)
            or stream['codec_name'] != 'h264'):
        raise ValueError(f'Encoded montage verification failed: {metadata}')
    for timestamp in (0, min(20, duration / 2), max(0, duration - 1)):
        image = output / f'page_{page:02d}_preview_{timestamp:.1f}s.png'
        run_logged([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(timestamp), '-i', str(video),
                    '-frames:v', '1', '-update', '1', str(image)], output / f'page_{page:02d}_preview.log')
    validation = {'file': video.name, 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'codec': stream['codec_name'],
                  'decoded_frames': int(stream['nb_read_frames']), 'declared_seconds': duration,
                  'encoded_seconds': float(metadata['format']['duration']), 'resolution': [width, height],
                  'page': page, 'total_evaluated_cases': report['total'], 'total_successes': report['successes'],
                  'stage': batch.name, 'runtime_sha256': report['runtime_sha256'], 'skill_sha256': report['skill_sha256'],
                  'source_cases': cases, 'labels_are_final_not_live': True, 'ended_clips_hold_final_frame': True,
                  'original_results_unchanged': True, 'artifact_integrity_passed': True}
    video.with_suffix('.validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'page': page, 'cases': len(cases), 'frames_verified': validation['decoded_frames']}), flush=True)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_root.resolve()
    if len(args.batch.split('/')) != 2 or not all(part.replace('-', '').isalnum() for part in args.batch.split('/')):
        raise ValueError('Expected round-name/batch-name')
    batch = ROOT / args.batch
    report = json.loads((batch / 'round_report.json').read_text(encoding='utf-8'))
    cases = sorted(report['results'], key=lambda row: (list(FAMILIES).index(row['family']), row['repeat']))
    for offset in range(0, len(cases), 12):
        selected = cases[offset:offset + 12]
        if len(selected) < 2:
            raise ValueError('A montage page requires at least two cases')
        create_page(batch, report, selected, offset // 12 + 1)


if __name__ == '__main__':
    main()
