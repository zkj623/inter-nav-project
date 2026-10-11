"""Unitree Go2 open-vocabulary semantic exploration on an unknown map."""

import argparse
import os
import sys
from datetime import datetime

from grutopia.core.util import has_display
from grutopia.demo.go2_point_navigation import build_objects
from grutopia_extension.configs.objects import (
    FixedCubeCfg,
    HouseholdSemanticPropsCfg,
    ProceduralHouseholdCfg,
)
from grutopia_extension.configs.robots.go2 import (
    DEFAULT_GO2_POLICY_PATH,
    DEFAULT_GO2_USD_PATH,
)
from grutopia_extension.interactive_navigation.go2_navigation_runner import (
    Go2SemanticExplorationRunConfig,
    run_go2_semantic_exploration,
)
from grutopia_extension.interactive_navigation.mapping_runtime import (
    SemanticDetectionMode,
)
from grutopia_extension.interactive_navigation.output_paths import resolve_map_output
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    load_point_navigation_profile,
    programmatic_go2_profile,
)

DEFAULT_GRSCENE_PROFILE = (
    'grutopia/demo/profiles/go2_grscene_mv7_exploration.json'
)

# Top surface of the flat collision floor laid over the scanned scene mesh.
# It must also be fed to the locomotion policy as the ground height, or the
# synthetic height scan reads the terrain 15 cm too low and Go2 walks in an
# unstable crouch.
GRSCENE_FLOOR_TOP = 0.15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable-root', help='private Kit data/cache/log root')
    parser.add_argument('--material-mode', choices=('original', 'preview', 'simple'), default=None)
    parser.add_argument('--target', default='refrigerator')
    parser.add_argument(
        '--scene',
        choices=('grscene', 'programmatic'),
        default='grscene',
    )
    parser.add_argument('--profile', default=DEFAULT_GRSCENE_PROFILE)
    parser.add_argument('--start', type=float, nargs=3, default=(0.0, 0.0, 0.40))
    parser.add_argument('--goal', type=float, nargs=3, default=(12.0, -2.6, 0.40))
    parser.add_argument('--environment-length', type=float, default=16.0)
    parser.add_argument('--environment-width', type=float, default=10.0)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--perception-gpu', type=int, default=5)
    parser.add_argument('--qwen-device', default='cuda:4')
    parser.add_argument('--max-steps', type=int, default=12000)
    parser.add_argument(
        '--verify-voronoi-incremental',
        action='store_true',
        help='smoke test: compare each successful seam splice with a full skeleton rebuild',
    )
    parser.add_argument('--mapping-warmup-steps', type=int, default=80)
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=not has_display(),
    )
    parser.add_argument(
        '--detection-mode',
        choices=tuple(mode.value for mode in SemanticDetectionMode),
        default=SemanticDetectionMode.OPEN_VOCABULARY.value,
        help=(
            'isaac: simulator ground-truth semantic labels only; '
            'open_vocab: GroundingDINO + MobileSAM detections only; '
            'hybrid: fuse both'
        ),
    )
    parser.add_argument('--semantic-classifier', choices=('qwen-vl', 'clip'), default='qwen-vl',
                        help='Qwen-VL classifies marked RGB crops; CLIP retains the legacy path.')
    parser.add_argument('--qwen-vl-url', default='http://localhost:12185/classify')
    parser.add_argument('--qwen-vl-timeout', type=float, default=60.0)
    parser.add_argument('--qwen-vl-max-candidates', type=int, default=12)
    parser.add_argument('--target-cues', action=argparse.BooleanOptionalAction, default=True,
                        help='Use DINO target proposals to approach and observe; never count them as class votes.')
    parser.add_argument(
        '--qwen',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--qwen-model', default='Qwen/Qwen3-8B')
    parser.add_argument(
        '--qwen-python',
        default=os.environ.get('QWEN3_PYTHON', 'python'),
    )
    parser.add_argument(
        '--map-output',
        default=None,
        help='map prefix; defaults to <record-dir>/final_map',
    )
    parser.add_argument(
        '--record-dir',
        default=None,
        help='output directory; defaults to a new timestamped directory',
    )
    parser.add_argument('--record-every', type=int, default=20)
    parser.add_argument('--video-fps', type=float, default=12.0)
    parser.add_argument('--rendering-interval', type=int, default=4)
    parser.add_argument(
        '--use-fabric',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--policy-path', default=DEFAULT_GO2_POLICY_PATH)
    parser.add_argument('--robot-usd', default=DEFAULT_GO2_USD_PATH)
    parser.add_argument(
        '--generate-fallback-asset',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_semantic_objects(args):
    """Keep the household floor, then add far refrigerator / chair / plant."""
    objects = []
    for obj in build_objects(args):
        if isinstance(obj, ProceduralHouseholdCfg):
            objects.append(obj.model_copy(update={'include_carry_object': False}))
        else:
            objects.append(obj)
    objects.append(
        HouseholdSemanticPropsCfg(
            name='household_semantic_props_0',
            prim_path='/World/env_0/objects/household_semantic_props',
        )
    )
    return objects


def build_grscene_floor(profile):
    """Give Go2 a flat collision plane above the scanned mesh floor.

    The 0.15 m top surface balances two constraints: low enough that lidar
    floor returns stay below the obstacle band (0.20 m) so couch seats keep
    stable occupancy, yet high enough to bury scan-mesh bumps and furniture
    plinths that the locomotion policy would otherwise climb and get stuck on
    (observed with a 0.10 m floor).
    """
    x_limits = profile.mapping.x_limits
    y_limits = profile.mapping.y_limits
    return [
        FixedCubeCfg(
            name='grscene_go2_floor_0',
            prim_path='/World/env_0/objects/grscene_go2_floor',
            position=(
                (x_limits[0] + x_limits[1]) / 2.0,
                (y_limits[0] + y_limits[1]) / 2.0,
                GRSCENE_FLOOR_TOP / 2.0,
            ),
            scale=(
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                GRSCENE_FLOOR_TOP,
            ),
            color=(0.42, 0.42, 0.44),
            semantic_label='floor',
        )
    ]


def build_profile(args):
    if args.scene == 'programmatic':
        return programmatic_go2_profile(
            environment_length=args.environment_length,
            environment_width=args.environment_width,
            start=tuple(args.start),
            goal=tuple(args.goal),
        )
    profile = load_point_navigation_profile(args.profile)
    if not profile.scene_asset_path:
        raise ValueError('GRScenes profile must define scene_asset_path')
    if not os.path.isfile(profile.scene_asset_path):
        raise FileNotFoundError(
            f'GRScenes navigation USD not found: {profile.scene_asset_path}'
        )
    return profile


def main():
    args = parse_args()
    record_dir = (
        args.record_dir if args.record_dir is not None
        else 'grutopia/results/go2_semantic_exploration_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    map_output = resolve_map_output(record_dir, args.map_output)
    profile = build_profile(args)
    objects = build_grscene_floor(profile) if args.scene == 'grscene' else build_semantic_objects(args)
    return run_go2_semantic_exploration(
        profile,
        Go2SemanticExplorationRunConfig(
            target_query=args.target,
            ground_height=GRSCENE_FLOOR_TOP if args.scene == 'grscene' else 0.0,
            gpu=args.gpu,
            perception_gpu=args.perception_gpu,
            qwen_device=args.qwen_device,
            headless=args.headless,
            max_steps=args.max_steps,
            mapping_warmup_steps=args.mapping_warmup_steps,
            record_dir=record_dir,
            record_every=args.record_every,
            video_fps=args.video_fps,
            map_output=map_output,
            policy_path=args.policy_path,
            robot_usd_path=args.robot_usd,
            generate_fallback_asset=args.generate_fallback_asset,
            semantic_detection_mode=args.detection_mode,
            semantic_classifier=args.semantic_classifier,
            qwen_vl_url=args.qwen_vl_url,
            qwen_vl_timeout=args.qwen_vl_timeout,
            qwen_vl_max_candidates=args.qwen_vl_max_candidates,
            enable_target_cues=args.target_cues,
            enable_qwen=args.qwen,
            qwen_model=args.qwen_model,
            qwen_python=args.qwen_python,
            rendering_interval=args.rendering_interval,
            use_fabric=args.use_fabric,
            material_mode=args.material_mode,
            verify_voronoi_incremental=args.verify_voronoi_incremental,
        ),
        objects=objects,
    )


if __name__ == '__main__':
    sys.exit(main())
