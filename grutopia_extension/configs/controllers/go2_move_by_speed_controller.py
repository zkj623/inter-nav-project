from typing import List, Optional

from grutopia.core.config.robot import ControllerCfg


class Go2MoveBySpeedControllerCfg(ControllerCfg):
    """Locomotion-policy configuration for the Unitree Go2."""

    type: Optional[str] = 'Go2MoveBySpeedController'
    joint_names: List[str]
    policy_weights_path: str
    # World z of the walking surface. The rough policy's synthetic height
    # scan is (base_z - ground_height - 0.5); on scenes with an elevated
    # collision floor a stale 0.0 makes the policy walk in a crouch.
    ground_height: float = 0.0
    actuator_mode: str = 'implicit_pd'
    stationary_joint_hold: bool = False
