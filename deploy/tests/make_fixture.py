"""Generate SYNTHETIC sensor input, not a simulator or physical acceptance run."""
import argparse
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.app.config import load, REPO
from deploy.robot.observation import Snapshot
from deploy.recording.recorder import Recorder


def make_fixture(root):
    recorder = Recorder(root, load(), 'synthetic_fixture_not_physics', REPO)
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    for index in range(20):
        x = min(index*.05, .85)
        now = 1000.+index*.1
        points = np.stack((4*np.cos(angles)+x, 4*np.sin(angles), np.full(720, .5)), axis=1).astype(np.float32)
        observation = {'position': np.array([x, 0, .3]), 'orientation': np.array([1., 0, 0, 0]),
                       'sensors': {'lidar': {'pointcloud': points, 'position': np.array([x, 0, .4]),
                                             'physics_step': index, 'max_range': 8.}}}
        recorder.observation(Snapshot(observation, now, now, now, None, (.05, 0, 0), index))
    recorder.close({'synthetic': True, 'physical_acceptance': False})
    return recorder.directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(make_fixture(parser.parse_args().output))
