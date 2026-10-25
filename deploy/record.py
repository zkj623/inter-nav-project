"""Record the current run.py session. Stopping this process never stops the robot."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    from deploy.recording.capture import record
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='http://127.0.0.1:8765', help='run.py session URL')
    parser.add_argument('--output', type=Path, help='local output root; default results/deploy')
    parser.add_argument('--wait', type=float, default=30., help='seconds to wait for run.py at startup')
    args = parser.parse_args()
    if not 0 <= args.wait <= 600:
        parser.error('--wait must be between 0 and 600')
    return record(args.source, args.output, args.wait)


if __name__ == '__main__':
    raise SystemExit(main())
