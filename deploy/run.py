"""Robot console. Default: observe only; replay never connects to the robot."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    # Resolve from this file, never from a developer's absolute path or shell cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    from deploy.app.config import load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--check', action='store_true', help='offline dependency/config check; no network or motion')
    parser.add_argument('--diagnose', action='store_true', help='read-only ROS receipt/interface diagnostic; no SDK')
    parser.add_argument('--duration', type=float, default=10., help='diagnostic subscription duration, 1..60 seconds')
    parser.add_argument('--allow-motion', action='store_true', help='allow explicit task keys to arm SportClient')
    parser.add_argument('--replay', type=Path, help='recorded run directory; hardware is never initialized')
    parser.add_argument('--task', choices=('observe', 'navigate'), default='observe', help='replay task only')
    parser.add_argument('--goal', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.diagnose and (args.allow_motion or args.replay or args.check):
        parser.error('--diagnose cannot be combined with motion, replay or offline check')
    if args.diagnose and not 1 <= args.duration <= 60:
        parser.error('--duration must be between 1 and 60 seconds')
    if args.replay and args.allow_motion:
        parser.error('--replay cannot be combined with --allow-motion')
    if args.task == 'navigate' and (not args.replay or args.goal is None):
        parser.error('--task navigate requires --replay and --goal X Y')
    try:
        config = load(args.config_dir)
        if args.diagnose:
            from deploy.robot.diagnostics import run
            return run(config, args.duration, args.output or Path('ros_diagnostic.json'))
        from deploy.app.runtime import check, run_live, run_replay
        if args.check:
            return check(config)
        if args.replay:
            return run_replay(args, config)
        return run_live(args, config)
    except (ValueError, ImportError, OSError, RuntimeError) as error:
        print(f'部署程序未就绪：{error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
