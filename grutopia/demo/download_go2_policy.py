"""Download and verify Unitree Go2 locomotion checkpoints from isaac-go2-ros2."""

import argparse
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from grutopia.macros import gm

POLICY_CATALOG = {
    'rough': {
        'filename': 'rough_model_7850.pt',
        'url': (
            'https://raw.githubusercontent.com/Zhefan-Xu/isaac-go2-ros2/'
            'isaacsim-4.5/ckpts/unitree_go2/rough_model_7850.pt'
        ),
        'size': 6881323,
        'sha256': (
            'd435f46dcf2d614cad28e21e033b34d6036d079924e1746530a36f2fba2315bd'
        ),
    },
    'flat': {
        'filename': 'flat_model_6800.pt',
        'url': (
            'https://raw.githubusercontent.com/Zhefan-Xu/isaac-go2-ros2/'
            'f5feb57dee24c408abe7bac41e81a62c3d5d567b/'
            'ckpts/unitree_go2/flat_model_6800.pt'
        ),
        'size': 983394,
        'sha256': (
            'd358d4a99a327f9c34f8538755655affd50cecf9a194a0ce0d374ba3228df8b0'
        ),
    },
}
DEFAULT_POLICY_NAME = 'rough'
DEFAULT_GO2_POLICY_DIR = (
    gm.ASSET_PATH + '/robots/go2/policy/move_by_speed'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--policy',
        choices=sorted(POLICY_CATALOG),
        default=DEFAULT_POLICY_NAME,
    )
    parser.add_argument('--output', default='')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def default_policy_path(policy: str = DEFAULT_POLICY_NAME) -> Path:
    return Path(DEFAULT_GO2_POLICY_DIR) / POLICY_CATALOG[policy]['filename']


def download_policy(
    output: str = '',
    force: bool = False,
    policy: str = DEFAULT_POLICY_NAME,
) -> Path:
    spec = POLICY_CATALOG[policy]
    output_path = Path(output).expanduser().resolve() if output else (
        default_policy_path(policy).expanduser().resolve()
    )
    if output_path.is_file() and not force:
        _validate(output_path, spec)
        print(f'Go2 {policy} policy already verified: {output_path}')
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        spec['url'],
        headers={'User-Agent': 'GRUtopia-Go2-policy-downloader/1.0'},
    )
    temporary_path = None
    try:
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.URLError as error:
            raise RuntimeError(
                'Could not download the Go2 policy. Check HTTPS/proxy access, '
                'or copy the checkpoint locally and pass --policy-path.'
            ) from error
        with response:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix='.go2-policy-',
                suffix='.tmp',
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary_file.write(chunk)
        _validate(temporary_path, spec)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(f'Downloaded verified Go2 {policy} policy: {output_path}')
    return output_path


def _validate(path: Path, spec: dict):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected_size = spec.get('size')
    expected_digest = spec.get('sha256')
    size_ok = not expected_size or len(data) == expected_size
    digest_ok = not expected_digest or digest == expected_digest
    if not size_ok or not digest_ok:
        raise RuntimeError(
            'Go2 policy verification failed: '
            f'bytes={len(data)} sha256={digest}'
        )


if __name__ == '__main__':
    arguments = parse_args()
    download_policy(
        arguments.output,
        force=arguments.force,
        policy=arguments.policy,
    )
