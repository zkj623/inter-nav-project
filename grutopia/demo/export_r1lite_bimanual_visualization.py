"""Export and fully decode the three verified lift videos for local viewing."""

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def validate_video(path, expected_frames, ffmpeg):
    probe = str(Path(ffmpeg).with_name('ffprobe' + Path(ffmpeg).suffix))
    completed = subprocess.run([probe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames',
                                '-of', 'json', str(path)], check=True, capture_output=True)
    stream = json.loads(completed.stdout)['streams'][0]
    fps = float(Fraction(stream['avg_frame_rate']))
    declared_frames = int(stream['nb_frames'])
    decoded = int(stream['nb_read_frames'])
    subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-nostdin', '-i', str(path),
                    '-map', '0:v:0', '-an', '-f', 'null', '-'], check=True, capture_output=True)
    samples = {}
    for index in (0, expected_frames // 2, expected_frames - 1):
        completed = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', str(path),
                                    '-vf', f'select=eq(n\\,{index})', '-fps_mode', 'passthrough',
                                    '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
                                   check=True, capture_output=True)
        frame = np.asarray(Image.open(io.BytesIO(completed.stdout)).convert('RGB'))
        if not np.isfinite(frame).all() or float(frame.std()) < 1.0:
            raise RuntimeError(f'Invalid frame {index} in {path}')
        samples[str(index)] = {'mean_pixel': float(frame.mean()), 'std_pixel': float(frame.std())}
    if decoded != expected_frames or declared_frames != expected_frames or abs(fps - 50.0) > 1e-3:
        raise RuntimeError(f'Incomplete video {path}: {decoded}/{expected_frames} at {fps} fps')
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'decoded_frames': decoded, 'declared_frames': declared_frames, 'fps': fps,
            'duration_seconds': decoded / fps, 'resolution': [stream['width'], stream['height']],
            'codec': stream['codec_name'], 'samples': samples, 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
    args = parser.parse_args()
    if not args.ffmpeg:
        raise RuntimeError('ffmpeg is required for compatible H.264 exports')
    result = json.loads((args.run / 'result.json').read_text())
    audit = json.loads((args.run / 'offline_reaudit.json').read_text())
    if not result['success'] or not result['collision_audit_passed'] or not audit['passed']:
        raise RuntimeError('Only a fully verified successful run may be exported')
    args.output.mkdir(parents=True, exist_ok=True)
    exports = (('bimanual_lift.mp4', '01_无穿模抱箱_全景.mp4'),
               ('bimanual_front.mp4', '02_无穿模抱箱_正面.mp4'),
               ('bimanual_side.mp4', '03_无穿模抱箱_侧面间隙.mp4'))
    validations = []
    for source_name, target_name in exports:
        source = args.run / source_name
        destination = args.output / target_name
        source_validation = validate_video(source, result['frames'], args.ffmpeg)
        if not destination.exists():
            subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                            '-i', str(source), '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)], check=True)
        validation = validate_video(destination, result['frames'], args.ffmpeg)
        validation['source_sha256'] = source_validation['sha256']
        validations.append(validation)
        print(f'Validated {destination.name}: {validation["decoded_frames"]} frames', flush=True)
    font = ImageFont.truetype(args.font, 28)
    comparison = Image.new('RGB', (1920, 780), (245, 245, 245))
    before = Image.open(args.run / '1_squeeze_side.jpg').convert('RGB')
    after = Image.open(args.run / '3_hold_side.jpg').convert('RGB')
    comparison.paste(before, (0, 60))
    comparison.paste(after, (960, 60))
    draw = ImageDraw.Draw(comparison)
    draw.text((24, 14), '夹持完成：箱子仍在地面', font=font, fill=(20, 20, 20))
    draw.text((984, 14), f'抬升完成：箱底约 {result["final_bottom_m"] * 100:.1f} cm，躯干未穿模',
              font=font, fill=(20, 20, 20))
    comparison.save(args.output / '04_侧视抬起前后对比.jpg', quality=95)
    shutil.copy2(args.run / '3_hold_side.jpg', args.output / '05_最终侧视间隙.jpg')
    shutil.copy2(args.run / '3_hold_close.jpg', args.output / '06_双手接触细节.jpg')
    for name in ('result.json', 'collision_audit.json', 'offline_reaudit.json', 'proxy_coverage.json'):
        shutil.copy2(args.run / name, args.output / name)
    (args.output / 'assets').mkdir(exist_ok=True)
    shutil.copy2(args.run / 'assets' / 'large_unhandled_box.usd', args.output / 'assets' / 'large_unhandled_box.usd')
    report = {'passed': all(entry['passed'] for entry in validations), 'videos': validations}
    (args.output / 'video_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
