from typing import List

import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.util import log
from grutopia_extension.configs.controllers import Go2MoveBySpeedControllerCfg
from grutopia_extension.controllers.isaac_go2_ctrl import Go2RSLControl


@BaseController.register('Go2MoveBySpeedController')
class Go2MoveBySpeedController(BaseController):
    """Velocity-command entry point for the robot's shared locomotion loop."""

    def __init__(
        self,
        config: Go2MoveBySpeedControllerCfg,
        robot: BaseRobot,
        scene: Scene,
    ) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        # Direct-speed and path controllers operate the same physical robot.
        # Share the low-level history, previous action and inference clock.
        contract = (config.policy_weights_path, tuple(config.joint_names), config.ground_height)
        existing = getattr(robot, '_go2_locomotion_control', None)
        if existing is None:
            robot._go2_locomotion_control = Go2RSLControl(
                config.policy_weights_path,
                self.robot.isaac_robot,
                config.joint_names,
                ground_height=config.ground_height,
            )
            robot._go2_locomotion_contract = contract
            control = robot._go2_locomotion_control
            log.info(
                f'Go2 loco: backend=rsl, '
                f'weights={config.policy_weights_path}, obs={control.observation_dim}, '
                'frequency=50Hz'
            )
        elif robot._go2_locomotion_contract != contract:
            raise ValueError('Go2 controllers on one robot must use the same locomotion configuration')
        self._control = robot._go2_locomotion_control
        self.joint_subset = self._control.joint_subset

    def forward(
        self,
        forward_speed: float = 0.0,
        rotation_speed: float = 0.0,
        lateral_speed: float = 0.0,
    ) -> ArticulationAction:
        self._control.set_command(forward_speed, lateral_speed, rotation_speed)
        return self._control.step()

    def action_to_control(
        self,
        action: List | np.ndarray,
    ) -> ArticulationAction:
        if len(action) != 3:
            raise ValueError('Go2 speed action must contain three elements')
        return self.forward(
            forward_speed=float(action[0]),
            lateral_speed=float(action[1]),
            rotation_speed=float(action[2]),
        )

    def get_obs(self):
        return self._control.get_obs()
