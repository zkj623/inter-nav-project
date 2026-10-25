"""JSON + numeric NPZ recordings: no pickle, no simulator objects."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import queue
import threading
from importlib.metadata import version, PackageNotFoundError
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def pack(value, arrays):
    if isinstance(value, np.ndarray):
        key = f'a{len(arrays)}'
        arrays[key] = value
        return {'__array__': key}
    if isinstance(value, dict):
        return {key: pack(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return arrays[value['__array__']].copy()
        return {key: unpack(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item, arrays) for item in value]
    return value


def provenance(repo):
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest = Path(repo) / '.deploy-source.json'
        source = json.loads(manifest.read_text()) if manifest.is_file() else {}
        commit, dirty = source.get('source_commit', 'unknown'), source.get('dirty_worktree')
    versions = {}
    for name in ('numpy', 'scipy', 'opencv-python-headless', 'PyYAML', 'requests', 'unitree-sdk2py'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"source_commit": commit, "dirty_worktree": dirty, "dependencies": versions}


class Recorder:
    def __init__(self, root, config, mode, repo, asynchronous=False, source=None):
        self.directory = Path(root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames = self.directory / 'observations'
        self.frames.mkdir()
        self.events = (self.directory / 'events.jsonl').open('w')
        self.index = (self.directory / 'observations.jsonl').open('w')
        self.count = 0
        self.mode = mode
        self.dropped = 0
        self.write_error = None
        self.pending = queue.Queue(maxsize=1) if asynchronous else None
        self.writer = None
        if asynchronous:
            self.writer = threading.Thread(target=self._write_loop, daemon=True)
            self.writer.start()
        source = source or provenance(repo)
        (self.directory / 'config.json').write_text(json.dumps(
            {'config': config, **source, 'mode': mode}, indent=2))

    def event(self, kind, **details):
        self.events.write(json.dumps({'event': kind, **details}, default=json_value, allow_nan=False)+'\n')
        self.events.flush()

    def observation(self, snapshot):
        if self.write_error:
            raise RuntimeError(self.write_error)
        if self.pending is not None:
            try:
                self.pending.put_nowait(snapshot)
            except queue.Full:
                self.dropped += 1
            return
        self._write(snapshot)

    def _write_loop(self):
        while True:
            snapshot = self.pending.get()
            if snapshot is None:
                break
            try:
                self._write(snapshot)
            except Exception as error:
                self.write_error = str(error)

    def _write(self, snapshot):
        arrays = {}
        metadata = pack(asdict(snapshot), arrays)
        filename = f'observations/{self.count:07d}.npz'
        np.savez_compressed(self.directory / filename, **arrays)
        self.index.write(json.dumps({'file': filename, 'snapshot': metadata}, allow_nan=False)+'\n')
        self.index.flush()
        self.count += 1

    def close(self, summary):
        if self.writer:
            self.pending.put(None)
            self.writer.join()
        if self.write_error:
            summary = {**summary, 'completed': False, 'recording_error': self.write_error}
        self.events.close()
        self.index.close()
        (self.directory / 'run_summary.json').write_text(json.dumps(
            {'mode': self.mode, 'observation_count': self.count, 'observation_dropped': self.dropped, **summary}, indent=2,
            default=json_value, allow_nan=False))
