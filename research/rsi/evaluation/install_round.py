import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    changes = []
    for source in sorted(args.runtime.rglob('*.py')):
        relative = source.relative_to(args.runtime)
        previous = args.baseline / relative
        destination = args.repo / relative
        if previous.exists() and checksum(source) == checksum(previous):
            continue
        if destination.exists() and (not previous.exists() or checksum(destination) != checksum(previous)):
            raise RuntimeError(f'Shared changes detected; will not overwrite {destination}')
        changes.append((source, destination, relative))
    for source, destination, relative in changes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    sys.path.insert(0, str(args.runtime))
    from grutopia_extension.interaction_clearance.benchmark import runtime_fingerprint

    manifest = runtime_fingerprint(args.runtime)
    (args.runtime.parent / 'runtime_manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.runtime.parent / 'installed_changes.json').write_text(json.dumps({
        str(relative): checksum(source) for source, destination, relative in changes}, indent=2))
    print(json.dumps({'installed': len(changes), 'runtime_sha256': manifest['sha256']}))


if __name__ == '__main__':
    main()
