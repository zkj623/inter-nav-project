"""Synthetic inputs passed through the real Tasks mapper; no hardware imports."""
import time
import cv2
import numpy as np
from deploy.robot.observation import Snapshot


class DemoSensors:
    def __init__(self, session):
        self.session, self.last_error = session, None
        self.step = 0
        angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        self.points = np.column_stack((4*np.cos(angles), 4*np.sin(angles), np.full(720, .5))).astype(np.float32)
        # Add a short obstacle face nearer than the outer walls.
        self.points = np.concatenate((self.points, np.column_stack((np.full(60, 1.5), np.linspace(.5, 1.3, 60), np.full(60, .5)))))
        # Ground samples make the calibrated ground/obstacle colors visible.
        x, y = np.meshgrid(np.arange(-3.75, 3.76, .25), np.arange(-3.75, 3.76, .25))
        floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        floor = floor[np.linalg.norm(floor[:, :2], axis=1) < 3.8]
        self.points = np.concatenate((self.points, floor))
        self.last_rgb = 0.

    def latest(self):
        now = time.monotonic()
        self.step += 1
        if now-self.last_rgb > .5:
            image = np.full((360, 640, 3), (28, 37, 47), dtype=np.uint8)
            cv2.rectangle(image, (80, 145), (560, 330), (80, 89, 98), -1)
            cv2.rectangle(image, (380, 155), (500, 285), (220, 170, 96), -1)
            cv2.putText(image, 'SYNTHETIC INPUT / NO ROBOT', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, (235, 235, 235), 2)
            cv2.putText(image, f'Frame {self.step}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (120, 220, 170), 1)
            self.session.rgb(image, now, 'Synthetic RGB (not RGB-D)')
            self.last_rgb = now
        observation = {'position': np.array([0., 0., .3]), 'orientation': np.array([1., 0., 0., 0.]),
                       'sensors': {'lidar': {'pointcloud': self.points, 'position': np.array([0., 0., .4]),
                                             'physics_step': self.step, 'max_range': 8.}}}
        return Snapshot(observation, now, now, now, None, (0., 0., 0.), self.step,
                        geometry_calibrated=True, planning_frame='demo_odom')

    def close(self):
        pass
