"""GRUtopia sensor wrapper for Isaac Sim's PhysX rotating LiDAR."""

from typing import Dict

import numpy as np
from omni.isaac.sensor import RotatingLidarPhysX
from omni.isaac.range_sensor import _range_sensor

from grutopia.core.robot.robot import BaseRobot, Scene
from grutopia.core.robot.sensor import BaseSensor
from grutopia_extension.configs.sensors import PhysXLidarCfg
from grutopia_extension.sensors.lidar_returns import exclude_robot_returns, freeze_lidar_scan


class _SynchronizedRotatingLidarPhysX(RotatingLidarPhysX):
    """Keep hit identities and pose beside the points acquired by Isaac."""

    def __init__(self, *args, **kwargs):
        self._synchronized_frame = {}
        super().__init__(*args, **kwargs)

    def _data_acquisition_callback(self, step_size):
        super()._data_acquisition_callback(step_size)
        if self.is_paused() or 'point_cloud' not in self._current_frame:
            return
        position, orientation = self.get_world_pose()
        self._synchronized_frame = freeze_lidar_scan(
            self._current_frame['point_cloud'],
            self._lidar_sensor_interface.get_prim_data(self.prim_path),
            position, orientation,
            self._current_frame['physics_step'], self._current_frame['time'],
        )

    def get_synchronized_frame(self):
        return self._synchronized_frame

    def post_reset(self):
        self._synchronized_frame = {}
        super().post_reset()


@BaseSensor.register('PhysXLidar')
class PhysXLidar(BaseSensor):
    """Expose a full-scan LiDAR point cloud in world coordinates."""

    def __init__(self, config: PhysXLidarCfg, robot: BaseRobot, name: str = None, scene: Scene = None):
        super().__init__(config, robot, scene)
        self._enabled = bool(config.enable)
        self._lidar = None
        if not self._enabled:
            return
        prim_path = f'{robot.config.prim_path}/{config.prim_path}'
        self._lidar = scene.add(
            _SynchronizedRotatingLidarPhysX(
                prim_path=prim_path,
                name=f'{name}_isaac',
                translation=np.asarray(config.translation, dtype=float),
                orientation=None if config.orientation is None else np.asarray(config.orientation, dtype=float),
                rotation_frequency=config.rotation_frequency,
                fov=config.fov,
                resolution=config.resolution,
                valid_range=config.valid_range,
            )
        )
        self._lidar.add_point_cloud_data_to_frame()
        # Remove only our own links, not nearby furniture or walls.
        self._lidar.enable_semantics()
        self._interface = _range_sensor.acquire_lidar_sensor_interface()
        self._robot_path = robot.config.prim_path

    def post_reset(self):
        if self._lidar is not None:
            self._lidar.post_reset()

    def get_data(self) -> Dict:
        if self._lidar is None:
            return {}
        frame = self._lidar.get_synchronized_frame()
        if not frame:
            return {}
        local_points = frame.get('point_cloud')
        local_points = _points_array(local_points)
        self_hits = 0
        if len(local_points):
            local_points, self_hits = exclude_robot_returns(
                local_points, frame['hit_paths'], self._robot_path
            )
        position, orientation = frame['position'], frame['orientation']
        world_points = _transform_points(local_points, position, orientation)
        return {
            'pointcloud': world_points,
            'local_pointcloud': local_points,
            'position': np.asarray(position, dtype=np.float32),
            'orientation': np.asarray(orientation, dtype=np.float32),
            'physics_step': int(frame.get('physics_step', 0)),
            'time': float(frame.get('time', 0.0)),
            'min_range': float(self.config.valid_range[0]),
            'max_range': float(self.config.valid_range[1]),
            'horizontal_fov': float(self.config.fov[0]),
            'rotation_frequency': float(self.config.rotation_frequency),
            'self_hits_removed': self_hits,
            'scan_pose_and_hit_paths_synchronized': True,
        }

    def cleanup(self):
        if self._lidar is not None:
            self._lidar.pause()


def _points_array(points) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return array.reshape(-1, 3)


def _transform_points(points: np.ndarray, position, quaternion) -> np.ndarray:
    if len(points) == 0:
        return points
    rotation = _quaternion_matrix(quaternion)
    return (points @ rotation.T + np.asarray(position, dtype=np.float32)).astype(np.float32)


def _quaternion_matrix(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
