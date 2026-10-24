"""Read-only point cloud / RGB viewer. Run alongside run.py in another terminal."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    from deploy.app.config import load
    from deploy.visualization.server import serve
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--camera', choices=('sdk', 'ros', 'none'), default='sdk',
                        help='sdk: Go2 front camera; ros: sensor_msgs/Image; none: cloud only')
    parser.add_argument('--rgb-topic', help='required with --camera ros unless robot.yaml sets rgb_topic')
    parser.add_argument('--fps', type=float, default=5., help='maximum acquisition/display rate, 0.5..10 Hz')
    parser.add_argument('--max-points', type=int, default=12000, help='display sampling limit, 100..40000')
    parser.add_argument('--demo', action='store_true', help='synthetic preview; no ROS, SDK or robot connection')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not .5 <= args.fps <= 10 or not 100 <= args.max_points <= 40000:
        parser.error('port: 1..65535; fps: 0.5..10; max-points: 100..40000')
    try:
        config = load(args.config_dir)['robot']
        topic = args.rgb_topic or config['rgb_topic']
        if args.camera == 'ros' and not topic and not args.demo:
            parser.error('--camera ros requires --rgb-topic /your/image/topic')
        return serve(config, args, topic)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f'可视化未就绪：{error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
