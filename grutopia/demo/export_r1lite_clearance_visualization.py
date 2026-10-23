"""Export verified clearance episodes as local H.264 videos and comparison images."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

from export_r1lite_bimanual_visualization import validate_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    audit = json.loads((args.run / 'offline_reaudit.json').read_text())
    if not args.ffmpeg or not result['success'] or not audit['passed']:
        raise RuntimeError('ffmpeg and a verified successful episode are required')
    args.output.mkdir(parents=True, exist_ok=True)
    validations = []
    for source_name, target_name in (('clearance_overview.mp4', '01_搬移清障_全景.mp4'),
                                     ('clearance_layout.mp4', '02_搬移清障_通道布局.mp4')):
        source = args.run / source_name
        destination = args.output / target_name
        original = validate_video(source, result['frames'], args.ffmpeg)
        if not destination.exists():
            subprocess.run([args.ffmpeg, '-v', 'error', '-nostdin', '-n', '-i', str(source), '-an',
                            '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(destination)], check=True)
        report = validate_video(destination, result['frames'], args.ffmpeg)
        report['source_sha256'] = original['sha256']
        validations.append(report)
        print(f'Validated {target_name}: {result["frames"]} frames', flush=True)
    comparison = Image.new('RGB', (1920, 780), (245, 245, 245))
    comparison.paste(Image.open(args.run / 'start_layout.jpg').convert('RGB'), (0, 60))
    comparison.paste(Image.open(args.run / 'settle_layout.jpg').convert('RGB'), (960, 60))
    font = ImageFont.truetype(args.font, 28)
    drawing = ImageDraw.Draw(comparison)
    drawing.text((24, 14), '开始：箱子堵住指定通行区域', font=font, fill=(20, 20, 20))
    drawing.text((984, 14), '结束：物体放稳并松手，通行区域清出', font=font, fill=(20, 20, 20))
    comparison.save(args.output / '03_清障前后对比.jpg', quality=95)
    shutil.copy2(args.run / 'release_layout.jpg', args.output / '04_放置并松手.jpg')
    for name in ('result.json', 'task.json', 'collision_audit.json', 'offline_reaudit.json',
                 'dataset_metadata.json', 'source_manifest.json', 'executed_skill.py', 'demonstration.npz'):
        shutil.copy2(args.run / name, args.output / name)
    (args.output / 'assets').mkdir(exist_ok=True)
    shutil.copy2(args.run / 'assets/large_unhandled_box.usd', args.output / 'assets/large_unhandled_box.usd')
    (args.output / 'video_validation.json').write_text(json.dumps({'passed': True, 'videos': validations},
                                                                indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
