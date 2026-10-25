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
    parser.add_argument('--output', type=Path, help='diagnostic/replay output only; live recording uses record.py')
    parser.add_argument('--session-port', type=int, default=8765, help='read-only data feed for view.py and record.py')
    parser.add_argument('--camera', choices=('sdk', 'ros', 'none'), default='sdk', help='preview RGB source; never substitutes for RGB-D')
    parser.add_argument('--rgb-topic', help='sensor_msgs/Image topic for --camera ros')
    parser.add_argument('--demo', action='store_true', help='synthetic inputs through the real mapper; no robot connection')
    parser.add_argument('--demo-seconds', type=float, default=0., help='optional demo time limit; 0 runs until Ctrl+C')
    args = parser.parse_args()
    if not 1 <= args.session_port <= 65535 or not 0 <= args.demo_seconds <= 86400:
        parser.error('invalid session port or demo duration')
    if args.demo and (args.allow_motion or args.diagnose or args.check or args.replay):
        parser.error('--demo cannot be combined with motion, diagnostics, check or replay')
    if args.output and not (args.diagnose or args.replay):
        parser.error('live output is managed by python record.py --output PATH')
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
        if args.camera == 'ros' and not args.demo and not (args.check or args.diagnose or args.replay):
            if not (args.rgb_topic or config['robot']['rgb_topic']):
                parser.error('--camera ros requires --rgb-topic or robot.yaml rgb_topic')
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
