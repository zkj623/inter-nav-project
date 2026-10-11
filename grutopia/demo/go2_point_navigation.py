"""Unitree Go2 RGB/LiDAR point navigation with synchronized review video."""

import argparse
import sys
from dataclasses import replace

from grutopia.core.util import has_display
from grutopia_extension.configs.objects import FixedCubeCfg, ProceduralHouseholdCfg
from grutopia_extension.configs.robots.go2 import (
    DEFAULT_GO2_POLICY_PATH,
    DEFAULT_GO2_USD_PATH,
)
from grutopia_extension.interactive_navigation.go2_navigation_runner import (
    Go2NavigationRunConfig,
    run_go2_point_navigation,
)
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    programmatic_go2_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=float, nargs=3, default=(0.0, 0.0, 0.40))
    parser.add_argument('--goal', type=float, nargs=3, default=(3.0, 0.0, 0.40))
    parser.add_argument('--environment-length', type=float, default=16.0)
    parser.add_argument('--environment-width', type=float, default=10.0)
    parser.add_argument('--voronoi-paths', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--success-distance', type=float, default=0.70)
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=not has_display(),
    )
    parser.add_argument(
        '--map-output',
        default='grutopia/results/go2_navigation/final_map',
    )
    parser.add_argument(
        '--record-dir',
        default='grutopia/results/go2_navigation',
    )
    parser.add_argument('--record-every', type=int, default=20)
    parser.add_argument('--video-fps', type=float, default=12.0)
    parser.add_argument('--mapping-warmup-steps', type=int, default=60)
    parser.add_argument('--policy-path', default=DEFAULT_GO2_POLICY_PATH)
    parser.add_argument('--robot-usd', default=DEFAULT_GO2_USD_PATH)
    parser.add_argument(
        '--generate-fallback-asset',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='generate the bundled mesh-free Go2 USD when --robot-usd is missing',
    )
    return parser.parse_args()


def build_objects(args):
    center_x = args.environment_length / 2.0 - 1.0
    half_width = args.environment_width / 2.0
    return [
        ProceduralHouseholdCfg(
            name='interactive_household_0',
            prim_path='/World/env_0/objects/interactive_household',
            floor_scale=(
                args.environment_length,
                args.environment_width,
                0.20,
            ),
            goal_position=(args.goal[0], args.goal[1], 0.02),
        ),
        FixedCubeCfg(
            name='far_obstacle_a',
            prim_path='/World/env_0/objects/far_obstacle_a',
            position=(6.5, 1.2, 0.60),
            scale=(0.70, 1.40, 1.20),
            color=(0.20, 0.45, 0.85),
        ),
        FixedCubeCfg(
            name='far_obstacle_b',
            prim_path='/World/env_0/objects/far_obstacle_b',
            position=(8.0, -1.6, 0.55),
            scale=(0.90, 0.90, 1.10),
            color=(0.80, 0.35, 0.15),
        ),
        FixedCubeCfg(
            name='far_obstacle_c',
            prim_path='/World/env_0/objects/far_obstacle_c',
            position=(9.4, 0.7, 0.75),
            scale=(0.60, 1.60, 1.50),
            color=(0.30, 0.70, 0.30),
        ),
        FixedCubeCfg(
            name='north_wall',
            prim_path='/World/env_0/objects/north_wall',
            position=(center_x, half_width, 1.0),
            scale=(args.environment_length, 0.20, 2.0),
            color=(0.65, 0.65, 0.68),
        ),
        FixedCubeCfg(
            name='south_wall',
            prim_path='/World/env_0/objects/south_wall',
            position=(center_x, -half_width, 1.0),
            scale=(args.environment_length, 0.20, 2.0),
            color=(0.65, 0.65, 0.68),
        ),
        FixedCubeCfg(
            name='east_wall',
            prim_path='/World/env_0/objects/east_wall',
            position=(args.environment_length - 1.0, 0.0, 1.0),
            scale=(0.20, args.environment_width, 2.0),
            color=(0.65, 0.65, 0.68),
        ),
        FixedCubeCfg(
            name='west_wall',
            prim_path='/World/env_0/objects/west_wall',
            position=(-1.0, 0.0, 1.0),
            scale=(0.20, args.environment_width, 2.0),
            color=(0.65, 0.65, 0.68),
        ),
    ]


def main():
    args = parse_args()
    profile = programmatic_go2_profile(
        environment_length=args.environment_length,
        environment_width=args.environment_width,
        start=tuple(args.start),
        goal=tuple(args.goal),
    )
    profile = replace(profile, success_distance=args.success_distance)
    return run_go2_point_navigation(
        profile,
        Go2NavigationRunConfig(
            gpu=args.gpu,
            prefer_voronoi_paths=args.voronoi_paths,
            headless=args.headless,
            max_steps=args.max_steps,
            mapping_warmup_steps=args.mapping_warmup_steps,
            record_dir=args.record_dir,
            record_every=args.record_every,
            video_fps=args.video_fps,
            map_output=args.map_output,
            policy_path=args.policy_path,
            robot_usd_path=args.robot_usd,
            generate_fallback_asset=args.generate_fallback_asset,
        ),
        objects=build_objects(args),
    )


if __name__ == '__main__':
    sys.exit(main())
