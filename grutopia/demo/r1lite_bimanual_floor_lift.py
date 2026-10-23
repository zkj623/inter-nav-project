"""Physically hug an unhandled floor box from both sides and lift it."""

import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--box-dimensions', type=float, nargs=3, default=(0.40, 0.50, 0.28))
    parser.add_argument('--box-mass', type=float, default=1.5)
    parser.add_argument('--grasp-height', type=float, default=0.25)
    parser.add_argument('--squeeze', type=float, default=0.003)
    parser.add_argument('--arm-stiffness', type=float, default=1200.0)
    parser.add_argument('--object-xy', type=float, nargs=2, default=(0.65, 0.0))
    parser.add_argument('--robot-usd')
    parser.add_argument('--robot-urdf')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise SystemExit('Use a new output directory.')
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': args.headless, 'active_gpu': args.gpu,
                         'physics_gpu': args.gpu, 'multi_gpu': False})
    success = False
    try:
        from grutopia_extension.manipulation.bimanual_floor_lift import BimanualBox
        from grutopia_extension.manipulation.r1lite_bimanual_lift_sim import R1LiteBimanualLift

        box = BimanualBox(tuple(args.box_dimensions), args.box_mass, args.grasp_height, args.squeeze)
        scene = R1LiteBimanualLift(output, box, args.object_xy, args.robot_usd, args.robot_urdf, args.arm_stiffness)
        result = scene.run()
        success = result['success']
        print(json.dumps(result), flush=True)
    except Exception as error:
        traceback.print_exc()
        (output / 'result.json').write_text(json.dumps({'success': False, 'error': str(error),
                                                      'training_performed': False}, indent=2))
    finally:
        app.close()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
