"""Record run.py, or capture read-only robot control traffic with --control."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true', help='read-only ROS control trace; starts without run.py')
    parser.add_argument('--duration', type=float, help='control trace seconds, 1..600; default 60')
    parser.add_argument('--source', default='http://127.0.0.1:8765', help='run.py session URL')
    parser.add_argument('--output', type=Path, help='local output root; default results/deploy')
    parser.add_argument('--wait', type=float, default=30., help='seconds to wait for run.py at startup')
    args = parser.parse_args()
    if args.control:
        duration = 60. if args.duration is None else args.duration
        if not 1 <= duration <= 600:
            parser.error('--duration must be between 1 and 600 seconds')
        if args.source != 'http://127.0.0.1:8765' or args.wait != 30.:
            parser.error('--control subscribes directly to ROS; do not use --source or --wait')
        from deploy.app.config import load
        from deploy.recording.control import record_control
        return record_control(load(), args.output, duration)
    if args.duration is not None:
        parser.error('--duration requires --control')
    if not 0 <= args.wait <= 600:
        parser.error('--wait must be between 0 and 600')
    from deploy.recording.capture import record
    return record(args.source, args.output, args.wait)


if __name__ == '__main__':
    raise SystemExit(main())
