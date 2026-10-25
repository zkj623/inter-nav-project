"""Constant-speed Go2 walk for gait review. No planner, no waypoint tracking."""

import argparse
import sys

from grutopia.core.util import has_display
from grutopia_extension.configs.objects import FixedCubeCfg
from grutopia_extension.configs.robots.go2 import (
    DEFAULT_GO2_POLICY_PATH,
    DEFAULT_GO2_USD_PATH,
)
from grutopia.demo.go2_semantic_exploration import (
    DEFAULT_GRSCENE_PROFILE,
    GRSCENE_FLOOR_TOP,
    build_grscene_floor,
)
from grutopia_extension.interactive_navigation.go2_navigation_runner import (
    Go2NavigationRunConfig,
    run_go2_constant_walk,
)
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    load_point_navigation_profile,
    programmatic_go2_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--scene',
        choices=('programmatic', 'grscene'),
        default='programmatic',
    )
    parser.add_argument('--profile', default=DEFAULT_GRSCENE_PROFILE)
    parser.add_argument('--start', type=float, nargs=3, default=(0.0, 0.0, 0.40))
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--warmup-steps', type=int, default=80)
    parser.add_argument(
        '--command',
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        help='body-frame [forward, lateral, yaw] sent to the policy every step',
    )
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=not has_display(),
    )
    parser.add_argument(
        '--record-dir',
        default='grutopia/results/go2_walk',
    )
    parser.add_argument(
        '--map-output',
        default='grutopia/results/go2_walk/final_map',
    )
    parser.add_argument('--record-every', type=int, default=10)
    parser.add_argument('--video-fps', type=float, default=12.0)
    parser.add_argument('--policy-path', default=DEFAULT_GO2_POLICY_PATH)
    parser.add_argument('--robot-usd', default=DEFAULT_GO2_USD_PATH)
    parser.add_argument(
        '--heading-hold',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='close a proportional yaw loop on the walk-start heading',
    )
    parser.add_argument('--heading-gain', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scene == 'grscene':
        profile = load_point_navigation_profile(args.profile)
        objects = build_grscene_floor(profile)
    else:
        profile = programmatic_go2_profile(
            environment_length=16.0,
            environment_width=10.0,
            start=tuple(args.start),
            goal=(12.0, 0.0, 0.40),
        )
        objects = [
            FixedCubeCfg(
                name='walk_floor',
                prim_path='/World/env_0/objects/walk_floor',
                position=(0.0, 0.0, -0.10),
                scale=(40.0, 40.0, 0.20),
                color=(0.55, 0.55, 0.58),
            ),
        ]
    return run_go2_constant_walk(
        profile,
        Go2NavigationRunConfig(
            gpu=args.gpu,
            headless=args.headless,
            max_steps=args.max_steps,
            mapping_warmup_steps=args.warmup_steps,
            record_dir=args.record_dir,
            record_every=args.record_every,
            video_fps=args.video_fps,
            map_output=args.map_output,
            policy_path=args.policy_path,
            robot_usd_path=args.robot_usd,
            ground_height=GRSCENE_FLOOR_TOP if args.scene == 'grscene' else 0.0,
        ),
        objects=objects,
        command=tuple(args.command),
        heading_hold=args.heading_hold,
        heading_gain=args.heading_gain,
    )


if __name__ == '__main__':
    sys.exit(main())
