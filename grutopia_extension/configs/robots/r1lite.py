"""Chinese Galaxea R1 Lite with three physical steering/drive modules."""

import os
from typing import Optional
from grutopia.core.config import RobotCfg
from grutopia.core.config.robot import ControllerCfg
from grutopia_extension.configs.controllers import MoveAlongPathPointsControllerCfg, MoveToPointBySpeedControllerCfg

DEFAULT_R1LITE_USD_PATH = os.environ.get(
    'R1LITE_USD_PATH', '/data20t/embodied/zkj/third_party/r1lite_isaac_v2/r1lite_navigation.usd'
)
R1LITE_STEER_JOINTS = tuple(f'steer_motor_joint{i}' for i in (1, 2, 3))
R1LITE_WHEEL_JOINTS = tuple(f'wheel_motor_joint{i}' for i in (1, 2, 3))
R1LITE_MODULE_POSITIONS = ((0.29183, -0.0014897), (-0.16817, 0.23851), (-0.16817, -0.24149))
# Relax the elbow bend and point the forearms/grippers down in front of the body.
R1LITE_ARM_HOME = (0.0, 3.02, -1.45, 0.0, 0.0, 0.0)
R1LITE_TORSO_HOME = (-0.95, 1.9, 0.95)
R1LITE_CAMERA_OFFSET = (0.13, 0.0, 1.12)
R1LITE_THIRD_PERSON_OFFSET = (-1.25, 0.0, 1.8)
R1LITE_THIRD_PERSON_PITCH_DEGREES = 44.0


class R1LiteSwerveControllerCfg(ControllerCfg):
    type: Optional[str] = 'R1LiteSwerveController'
    wheel_radius: float = 0.07
    max_wheel_speed: float = 12.0


class R1LiteRobotCfg(RobotCfg):
    name: Optional[str] = 'r1lite'
    type: Optional[str] = 'R1LiteRobot'
    prim_path: Optional[str] = '/r1lite'
    create_robot: Optional[bool] = True
    usd_path: Optional[str] = DEFAULT_R1LITE_USD_PATH


def r1lite_navigation_controller_cfgs():
    return (
        R1LiteSwerveControllerCfg(name='move_by_speed'),
        MoveAlongPathPointsControllerCfg(
            name='move_along_path',
            forward_speed=0.5,
            rotation_speed=0.6,
            threshold=0.12,
            sub_controllers=[
                MoveToPointBySpeedControllerCfg(
                    name='move_to_point',
                    forward_speed=0.5,
                    rotation_speed=0.6,
                    threshold=0.08,
                    sub_controllers=[R1LiteSwerveControllerCfg(name='move_by_speed')],
                )
            ],
        ),
    )
