import json
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2:
        raise ValueError('Expected simulation script and its arguments')
    ipc = Path('/dev/shm') / f'carb-RStringInternals-{os.getpid()}'
    if os.name == 'posix' and ipc.exists():
        print(json.dumps({'event': 'SDK_IPC_NAME_COLLISION', 'pid': os.getpid(), 'ipc': str(ipc),
                          'physics_started': False, 'foreign_objects_removed': False}), flush=True)
        return 75
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
