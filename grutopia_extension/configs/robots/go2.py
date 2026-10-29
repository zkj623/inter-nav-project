from typing import Optional, Literal

from grutopia.core.config import RobotCfg
from grutopia.macros import gm
from grutopia_extension.configs.controllers import (
    Go2MoveBySpeedControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
)

GO2_JOINT_NAMES = [
    'FL_hip_joint',
    'FR_hip_joint',
    'RL_hip_joint',
    'RR_hip_joint',
    'FL_thigh_joint',
    'FR_thigh_joint',
    'RL_thigh_joint',
    'RR_thigh_joint',
    'FL_calf_joint',
    'FR_calf_joint',
    'RL_calf_joint',
    'RR_calf_joint',
]

DEFAULT_GO2_POLICY_PATH = (
    gm.ASSET_PATH + '/robots/go2/policy/move_by_speed/rough_model_7850.pt'
)
DEFAULT_GO2_USD_PATH = gm.ASSET_PATH + '/robots/go2/isaaclab_go2.usd'
OFFICIAL_GO2_USD_URL = (
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/4.2/Isaac/Robots/Unitree/Go2/go2.usd'
)


def go2_navigation_controller_cfgs(
    policy_weights_path: str = DEFAULT_GO2_POLICY_PATH,
    ground_height: float = 0.0,
    actuator_mode: str = 'implicit_pd',
):
    """Build independent direct-speed and path-following controller trees."""

    move_by_speed = Go2MoveBySpeedControllerCfg(
        name='move_by_speed',
        policy_weights_path=policy_weights_path,
        joint_names=GO2_JOINT_NAMES,
        ground_height=ground_height,
        actuator_mode=actuator_mode,
    )
    nested_move_by_speed = Go2MoveBySpeedControllerCfg(
        name='move_by_speed',
        policy_weights_path=policy_weights_path,
        joint_names=GO2_JOINT_NAMES,
        ground_height=ground_height,
        actuator_mode=actuator_mode,
    )
    move_to_point = MoveToPointBySpeedControllerCfg(
        name='move_to_point',
        forward_speed=0.80,
        rotation_speed=0.8,
        threshold=0.08,
        sub_controllers=[nested_move_by_speed],
    )
    move_along_path = MoveAlongPathPointsControllerCfg(
        name='move_along_path',
        forward_speed=0.80,
        rotation_speed=0.8,
        threshold=0.12,
        sub_controllers=[move_to_point],
    )
    return move_by_speed, move_along_path


move_by_speed_cfg, move_along_path_cfg = go2_navigation_controller_cfgs()


class Go2RobotCfg(RobotCfg):
    name: Optional[str] = 'go2'
    type: Optional[str] = 'Go2Robot'
    prim_path: Optional[str] = '/go2'
    create_robot: Optional[bool] = True
    usd_path: Optional[str] = DEFAULT_GO2_USD_PATH
    generate_fallback_asset: bool = True
    physics_profile: Literal["legacy", "isaaclab_training"] = "legacy"
