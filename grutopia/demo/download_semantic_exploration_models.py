"""Download local models used by semantic exploration."""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


MOBILE_SAM_URL = (
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument(
        '--mobile-sam-output',
        default='grutopia/assets/models/mobile_sam.pt',
    )
    parser.add_argument('--skip-qwen', action='store_true')
    parser.add_argument('--skip-clip', action='store_true')
    parser.add_argument('--skip-grounding-dino', action='store_true')
    parser.add_argument('--skip-mobile-sam', action='store_true')
    return parser.parse_args()


def _download_huggingface(model_id: str, cache_dir):
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f'{model_id}: {path}', flush=True)


def _download_mobile_sam(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    urllib.request.urlretrieve(MOBILE_SAM_URL, temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'MobileSAM: {path} ({path.stat().st_size} bytes, sha256={digest})', flush=True)


def main():
    args = parse_args()
    if not args.skip_qwen:
        _download_huggingface('Qwen/Qwen3-8B', args.cache_dir)
    if not args.skip_clip:
        _download_huggingface('openai/clip-vit-base-patch32', args.cache_dir)
    if not args.skip_grounding_dino:
        _download_huggingface('IDEA-Research/grounding-dino-base', args.cache_dir)
    if not args.skip_mobile_sam:
        _download_mobile_sam(Path(args.mobile_sam_output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
