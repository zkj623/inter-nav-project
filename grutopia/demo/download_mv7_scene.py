"""Download the GRScenes MV7 home used by Go2 semantic exploration."""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from grutopia.macros import gm

SCENE_ID = 'MV7J6NIKTKJZ2AABAAAAADA8_usd'
DATASET_REPO = 'OpenRobotLab/GRScenes'
SCENE_RELATIVE = Path('scenes/GRScenes-100/home_scenes/scenes') / SCENE_ID
META_RELATIVE = Path('benchmark/meta') / SCENE_ID


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', default=gm.ASSET_PATH)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _looks_complete(scene_dir: Path, meta_dir: Path) -> bool:
    return (scene_dir / 'start_result_navigation.usd').is_file() and (
        meta_dir / 'object_dict.json'
    ).is_file()


def _extract_matching_zips(root: Path):
    for zip_path in root.rglob('*.zip'):
        if SCENE_ID not in zip_path.name and SCENE_ID not in str(zip_path.parent):
            continue
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)
        print(f'extracted {zip_path}', flush=True)


def _promote_extracted_scene(root: Path, destination: Path):
    if destination.exists():
        return
    for candidate in root.rglob(SCENE_ID):
        if candidate.is_dir() and (candidate / 'start_result_navigation.usd').is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, destination)
            print(f'copied scene to {destination}', flush=True)
            return


def main():
    args = parse_args()
    asset_root = Path(args.asset_root).expanduser().resolve()
    scene_dir = asset_root / SCENE_RELATIVE
    meta_dir = asset_root / META_RELATIVE
    if _looks_complete(scene_dir, meta_dir) and not args.force:
        print(f'MV7 scene already present: {scene_dir}')
        print(f'MV7 metadata already present: {meta_dir}')
        return 0

    from huggingface_hub import snapshot_download

    asset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type='dataset',
        local_dir=str(asset_root),
        allow_patterns=[
            f'**/{SCENE_ID}/**',
            f'**/{SCENE_ID}*',
            f'**/{SCENE_ID}.zip',
            '**/home_scenes*.zip',
        ],
    )
    _extract_matching_zips(asset_root)
    _promote_extracted_scene(asset_root, scene_dir)
    if not _looks_complete(scene_dir, meta_dir):
        raise FileNotFoundError(
            'Could not place the MV7 navigation USD and object_dict.json. '
            'Download scenes/GRScenes-100 and benchmark/meta for '
            f'{SCENE_ID} from https://huggingface.co/datasets/OpenRobotLab/GRScenes '
            f'into {asset_root}.'
        )
    print(f'MV7 scene ready: {scene_dir}')
    print(f'MV7 metadata ready: {meta_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
