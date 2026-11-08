"""Create a small floor scene and physically lift a box or a bin."""

import argparse
import json
from pathlib import Path
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=('box', 'bin'), default='box')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--object-xy', type=float, nargs=2, default=(0.60, -0.335))
    parser.add_argument('--robot-usd')
    parser.add_argument('--robot-urdf')
    parser.add_argument('--object-usd')
    parser.add_argument('--grasp-offset', type=float, nargs=3)
    parser.add_argument('--object-dimensions', type=float, nargs=3)
    parser.add_argument('--object-mass', type=float)
    parser.add_argument('--policy')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise SystemExit('Use a new output directory; existing results will not be overwritten.')
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': args.headless, 'active_gpu': args.gpu,
                         'physics_gpu': args.gpu, 'multi_gpu': False})
    success = False
    try:
        from grutopia_extension.manipulation.r1lite_lift_sim import R1LiteFloorLift

        policy = None
        if args.policy:
            from grutopia_extension.manipulation.lift_policy import load_policy

            policy = load_policy(args.policy)
        scene = R1LiteFloorLift(
            output, args.object, args.object_xy, args.robot_usd, args.robot_urdf,
            object_usd=args.object_usd, grasp_offset=args.grasp_offset,
            object_dimensions=args.object_dimensions, object_mass=args.object_mass,
        )
        report = scene.run(policy)
        success = report['success']
        print(json.dumps(report), flush=True)
    except Exception as error:
        traceback.print_exc()
        (output / 'result.json').write_text(json.dumps({'success': False, 'error': str(error)}, indent=2))
    finally:
        app.close()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
