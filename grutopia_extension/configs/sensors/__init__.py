from typing import Optional, Tuple

from grutopia.core.config.robot import SensorCfg


class CameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = 'Camera'
    enable: Optional[bool] = True
    resolution: Optional[Tuple[int, int]] = None
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None


class RepCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = 'RepCamera'
    enable: Optional[bool] = True
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    depth: Optional[bool] = False


class MocapControlledCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = 'MocapControlledCamera'
    enable: Optional[bool] = True
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None  # Quaternion in local frame


class PhysXLidarCfg(SensorCfg):
    type: Optional[str] = 'PhysXLidar'
    enable: Optional[bool] = True
    translation: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.25)
    orientation: Optional[Tuple[float, float, float, float]] = None
    rotation_frequency: Optional[float] = 0.0
    fov: Optional[Tuple[float, float]] = (360.0, 30.0)
    resolution: Optional[Tuple[float, float]] = (2.0, 5.0)
    valid_range: Optional[Tuple[float, float]] = (0.25, 8.0)
