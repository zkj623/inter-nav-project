"""Display RGB, the actual navigation map and its cumulative cloud from run.py."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    from deploy.visualization.server import serve
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='http://127.0.0.1:8765', help='run.py read-only session URL')
    parser.add_argument('--port', type=int, default=8080, help='browser UI port (loopback only)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    try:
        return serve(args)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f'可视化未就绪：{error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
