"""Download the official Unitree Go2 USD through Isaac Sim's client."""

import argparse
import sys
from pathlib import Path

from grutopia_extension.configs.robots.go2 import DEFAULT_GO2_USD_PATH
from grutopia_extension.robots.go2_asset import (
    OFFICIAL_GO2_USD_CANDIDATES,
    _is_meshed_usd,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_GO2_USD_PATH)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            'headless': True,
            'active_gpu': args.gpu,
            'physics_gpu': args.gpu,
        }
    )
    try:
        import omni.client

        for url in OFFICIAL_GO2_USD_CANDIDATES:
            print(f'trying {url}', flush=True)
            result = omni.client.copy(url, str(output_path))
            print(f'omni.client.copy -> {result}', flush=True)
            if _is_meshed_usd(output_path):
                print(f'downloaded official Go2 USD: {output_path} ({output_path.stat().st_size} bytes)')
                return 0
        raise RuntimeError(
            'Isaac Sim could not copy the official Unitree Go2 USD. '
            'Place go2.usd or the official dae meshes under the robot asset directory.'
        )
    finally:
        simulation_app.close()


if __name__ == '__main__':
    sys.exit(main())
