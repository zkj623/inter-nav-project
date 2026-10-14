"""Reusable camera rig that feeds both mapping and first-person review."""

from typing import Optional, Tuple

from grutopia_extension.interactive_navigation.visualization import FollowingRobotCamera


# Look forward from the original G1 head mount. Isaac/USD cameras default to a
# 1 m near plane, which turns nearby furniture into a black slab; keep 1 cm.
G1_FIRST_PERSON_OFFSET = (0.14, 0.0, 0.48)
G1_FIRST_PERSON_PITCH_DEGREES = 3.0
G1_FIRST_PERSON_CLIPPING_RANGE = (0.01, 80.0)
# Official Go2 head sits near x=0.29; keep the RGB camera past the snout.
GO2_FIRST_PERSON_OFFSET = (0.42, 0.0, 0.14)
GO2_FIRST_PERSON_PITCH_DEGREES = 6.0


class NavigationSensorRig:
    """Inject a stabilized head-height RGB-D observation into robot observations."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 360),
        offset=G1_FIRST_PERSON_OFFSET,
        pitch_degrees: float = G1_FIRST_PERSON_PITCH_DEGREES,
        horizontal_fov_degrees: float = 96.0,
        clipping_range=G1_FIRST_PERSON_CLIPPING_RANGE,
        rgb_interval: int = 24,
    ):
        if rgb_interval <= 0:
            raise ValueError('rgb_interval must be positive')
        self.rgb_interval = rgb_interval
        self.camera = FollowingRobotCamera(
            resolution=resolution,
            offset=offset,
            pitch_degrees=pitch_degrees,
            horizontal_fov_degrees=horizontal_fov_degrees,
            clipping_range=clipping_range,
            enable_depth=True,
        )
        self._cached_data: Optional[dict] = None
        self.frames = 0

    def prime(self, robot_observation: dict):
        """Place the camera before a simulation render produces its first frame."""

        position = robot_observation.get('head_position', robot_observation['position'])
        self.camera.update_pose(position, robot_observation['orientation'])

    def update(
        self,
        step: int,
        robot_observation: dict,
        force_capture: bool = False,
    ) -> dict:
        # Read the frame before changing the prim pose: depth, RGB, point cloud,
        # and the reported camera transform then describe the same rendered pose.
        if force_capture or self._cached_data is None or step % self.rgb_interval == 0:
            self._cached_data = self.camera.get_data()
            self._cached_data['capture_step'] = step
            if self._cached_data.get('depth') is not None:
                self.frames += 1
        robot_observation.setdefault('sensors', {})['camera'] = self._cached_data
        position = robot_observation.get('head_position', robot_observation['position'])
        # Stable body yaw deliberately excludes gait roll/pitch. Each robot
        # runner supplies the calibrated sensor-frame offset.
        self.camera.update_pose(position, robot_observation['orientation'])
        return self._cached_data
