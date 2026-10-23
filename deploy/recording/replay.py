"""Read recorded observations; this module has no robot communication path."""
import json
from pathlib import Path
import numpy as np
from deploy.robot.observation import Snapshot
from .recorder import unpack


def snapshots(directory):
    root = Path(directory).resolve()
    previous = None
    with (root / 'observations.jsonl').open() as stream:
        for line in stream:
            entry = json.loads(line)
            path = (root / entry['file']).resolve()
            if root not in path.parents:
                raise ValueError('Recording array path escapes run directory')
            with np.load(path, allow_pickle=False) as arrays:
                snapshot = Snapshot(**unpack(entry['snapshot'], arrays))
            if previous is not None and snapshot.captured_at <= previous:
                raise ValueError('Recording timestamps must increase')
            previous = snapshot.captured_at
            yield snapshot
