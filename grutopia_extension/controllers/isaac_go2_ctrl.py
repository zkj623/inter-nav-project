"""Go2 RSL-RL observation and motor interface for InternUtopia.

The actor uses the original checkpoint, 235 ordered observations, action
scale 0.25 and a 50 Hz policy clock. ``explicit_dc`` recalculates the trained
25/0.5 PD motor and its 23.5 N m torque-speed envelope before EVERY 5 ms
physics step; it rejects batched physics advances. ``implicit_pd`` uses
native PhysX position drives and is retained for legacy reproductions.
Native per-joint speed constraints remain those authored in the Go2 USD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.utils.types import ArticulationAction

from grutopia_extension.controllers.go2_policy import (
    DEFAULT_ACTION_SCALE,
    angular_velocity_from_quaternions,
    build_go2_observation,
    load_go2_actor,
    rotate_vector_inverse_wxyz,
    yaw_from_wxyz,
)
from grutopia_extension.configs.robots.go2 import GO2_JOINT_NAMES
from grutopia_extension.robots.go2_asset import GO2_DEFAULT_JOINT_POSITIONS
from grutopia_extension.interactive_navigation.go2_actuation import dc_motor_efforts

# Isaac Lab trained Go2 at 50 Hz with dt=0.005. Their play cfg uses 40 Hz.
POLICY_DECIMATION = 4
STIFFNESS = 25.0
DAMPING = 0.5
SATURATION_EFFORT = 23.5
VELOCITY_LIMIT = 30.0
COMMAND_CLIP = ((-1.5, -1.0, -1.5), (1.5, 1.0, 1.5))

base_vel_cmd_input = None


def init_base_vel_cmd(num_envs: int = 1) -> torch.Tensor:
    """Create the external velocity-command buffer used by Go2RSLEnvCfg."""

    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((int(num_envs), 3), dtype=torch.float32)
    return base_vel_cmd_input


def set_base_vel_cmd(command: Iterable[float], env_index: int = 0) -> torch.Tensor:
    """Write one ``[vx, vy, yaw]`` command, matching ``go2_ctrl.base_vel_cmd``."""

    if base_vel_cmd_input is None:
        init_base_vel_cmd(env_index + 1)
    clipped = np.clip(
        np.asarray(command, dtype=np.float32).reshape(3),
        COMMAND_CLIP[0],
        COMMAND_CLIP[1],
    )
    base_vel_cmd_input[env_index] = torch.as_tensor(clipped, dtype=torch.float32)
    return base_vel_cmd_input[env_index]


def base_vel_cmd(env_index: int = 0) -> np.ndarray:
    """Return the current command row as a numpy vector."""

    if base_vel_cmd_input is None:
        init_base_vel_cmd()
    return base_vel_cmd_input[env_index].detach().cpu().numpy().astype(np.float32)


class Go2RSLControl:
    """Shared locomotion control loop for one InternUtopia Go2."""

    def __init__(
        self,
        policy_weights_path: str,
        articulation,
        joint_names: Iterable[str],
        ground_height: float = 0.0,
        actuator_mode: str = 'implicit_pd',
        stationary_joint_hold: bool = False,
    ) -> None:
        init_base_vel_cmd(1)
        if actuator_mode not in ['implicit_pd','explicit_dc']:raise ValueError('Invalid Go2 actuator mode')
        if stationary_joint_hold and actuator_mode!='explicit_dc':
            raise ValueError('Stationary joint holding requires explicit_dc')
        self.actuator_mode=actuator_mode
        self._last_control_time=None
        self.applied_joint_efforts=np.zeros(12,dtype=np.float32)
        self.ground_height = float(ground_height)
        from grutopia_extension.interactive_navigation.go2_stationary_hold import StationaryJointHold
        self.stationary_hold_control = StationaryJointHold(ground_height=self.ground_height) if stationary_joint_hold else None
        self.stationary_hold_allowed = True
        joint_names = list(joint_names)
        self.policy = load_go2_actor(policy_weights_path)
        self.policy_command = np.zeros(3, dtype=np.float32)
        self.observation_dim = int(self.policy[0].in_features)
        self.joint_subset = ArticulationSubset(articulation, joint_names)
        self.articulation = articulation
        self.default_joint_positions = np.asarray(
            [GO2_DEFAULT_JOINT_POSITIONS[GO2_JOINT_NAMES.index(name)] for name in joint_names],
            dtype=np.float32,
        )
        self.last_action = np.zeros(12, dtype=np.float32)
        self.raw_policy_action = np.zeros(12, dtype=np.float32)
        self.applied_joint_positions = self.default_joint_positions.copy()
        self.apply_times_left = 0
        self.policy_name = Path(policy_weights_path).name
        self.body_lin_vel = np.zeros(3, dtype=np.float32)
        self.body_ang_vel = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self._prev_position = None
        self._prev_orientation = None
        self._last_inference_time = None
        self.policy_inferences = 0

    def reset(self):
        """Discard policy and timing state when the physical robot resets."""
        self.policy_command.fill(0)
        self._last_control_time=None
        if hasattr(self,'applied_joint_efforts'):self.applied_joint_efforts.fill(0.)
        self.last_action.fill(0)
        if hasattr(self,'raw_policy_action'):self.raw_policy_action.fill(0.)
        if getattr(self,'stationary_hold_control',None) is not None:self.stationary_hold_control.reset()
        self.stationary_hold_allowed=True
        self.applied_joint_positions = self.default_joint_positions.copy()
        self.apply_times_left = 0
        self._prev_position = None
        self._prev_orientation = None
        self._last_inference_time = None
        self.policy_inferences = 0
        self.body_lin_vel.fill(0)
        self.body_ang_vel.fill(0)
        self.yaw = 0.0
        set_base_vel_cmd((0., 0., 0.))

    def set_command(
        self,
        forward_speed: float,
        lateral_speed: float,
        rotation_speed: float,
    ) -> np.ndarray:
        command = np.asarray((forward_speed, lateral_speed, rotation_speed))
        if not np.isfinite(command).all():
            raise ValueError('Go2 velocity command must be finite')
        return set_base_vel_cmd(command).numpy()

    def step(self) -> ArticulationAction:
        now=self._simulation_time()
        previous=getattr(self,'_last_control_time',None)
        if getattr(self,'actuator_mode','implicit_pd')=='explicit_dc' and now is not None and previous is not None:
            if now-previous>.005+1e-6:
                raise RuntimeError('Explicit Go2 DC control requires one 5 ms physics step per actuator update')
        self._last_control_time=now
        if self._should_infer():
            self._infer_policy()
        if getattr(self,'actuator_mode','implicit_pd')=='explicit_dc':
            self.applied_joint_efforts=dc_motor_efforts(self.applied_joint_positions,
                self.joint_subset.get_joint_positions(),self.joint_subset.get_joint_velocities(),
                stiffness=STIFFNESS,damping=DAMPING,saturation_effort=SATURATION_EFFORT,
                effort_limit=SATURATION_EFFORT,velocity_limit=VELOCITY_LIMIT)
            return ArticulationAction(joint_efforts=self.applied_joint_efforts.copy(),
                                      joint_indices=self.joint_subset.joint_indices)
        # Position targets against the implicit PhysX drive: the drive
        # recomputes PD torque at every physics substep while the target is
        # held, including physics steps advanced during rendering.
        return ArticulationAction(
            joint_positions=self.applied_joint_positions.copy(),
            joint_indices=self.joint_subset.joint_indices,
        )

    def _should_infer(self) -> bool:
        """Run the policy every POLICY_DECIMATION * 0.005 s of simulation time.

        env.step calls advance 5 ms normally but 20 ms on render steps, so
        counting calls would give an irregular 20-35 ms policy period. The
        simulation clock keeps it at the exact 20 ms cadence the policy was
        trained with. Without a running simulation (unit tests) it falls back
        to counting calls.
        """

        now = self._simulation_time()
        if now is None:
            if self.apply_times_left > 0:
                self.apply_times_left -= 1
                return False
            self.apply_times_left = POLICY_DECIMATION - 1
            return True
        period = POLICY_DECIMATION * 0.005
        if self._last_inference_time is not None and now < self._last_inference_time:
            self.reset()
        if self._last_inference_time is None or now - self._last_inference_time >= period - 1e-6:
            self._last_inference_time = now
            return True
        return False

    @staticmethod
    def _simulation_time():
        try:
            from omni.isaac.core.simulation_context import SimulationContext
        except Exception:
            return None
        context = SimulationContext.instance()
        if context is None:
            return None
        return float(context.current_time)

    def get_obs(self) -> dict:
        return {
            'simulation_time': self._simulation_time(),
            'last_inference_time': self._last_inference_time,
            'policy_inferences': self.policy_inferences,
            'command': base_vel_cmd(),
            'policy_command': self.policy_command.copy(),
            'policy_action': self.last_action.copy(),
            'raw_policy_action': getattr(self,'raw_policy_action',self.last_action).copy(),
            'motor_joint_target': self.applied_joint_positions.copy(),
            'stationary_hold': (self.stationary_hold_control.report() if getattr(self,'stationary_hold_control',None) is not None else None),
            'policy_name': self.policy_name,
            'observation_dim': self.observation_dim,
            'policy_backend': 'rsl',
            'actuator_mode':getattr(self,'actuator_mode','implicit_pd'),
            'body_lin_vel': self.body_lin_vel.copy(),
            'body_ang_vel': self.body_ang_vel.copy(),
            'yaw': self.yaw,
            'ground_height': self.ground_height,
            'source': 'isaac-go2-ros2/Go2RSLEnvCfg',
        }

    def _root_state(self):
        position, orientation = self.articulation.get_world_pose()
        position = np.asarray(position, dtype=np.float32).reshape(3)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(4)
        linear_world, angular_world = self._root_velocities(position, orientation)
        self._prev_position = position.copy()
        self._prev_orientation = orientation.copy()
        linear_body = rotate_vector_inverse_wxyz(orientation, linear_world)
        angular_body = rotate_vector_inverse_wxyz(orientation, angular_world)
        gravity_body = rotate_vector_inverse_wxyz(orientation, (0.0, 0.0, -1.0))
        self.body_lin_vel = np.asarray(linear_body, dtype=np.float32)
        self.body_ang_vel = np.asarray(angular_body, dtype=np.float32)
        self.yaw = yaw_from_wxyz(orientation)
        return position, self.body_lin_vel, self.body_ang_vel, gravity_body

    def _root_velocities(self, position, orientation):
        """World-frame root velocities, preferring the PhysX ground truth.

        Isaac Lab's ``base_lin_vel``/``base_ang_vel`` observations read the
        simulator's root velocities directly; pose differencing over a policy
        period lags by half a step and aliases contact impacts, so it remains
        only as a fallback for contexts without a stepping simulation.
        """

        try:
            linear = self.articulation.get_linear_velocity()
            angular = self.articulation.get_angular_velocity()
        except Exception:
            linear = None
            angular = None
        if linear is not None and angular is not None:
            return (
                np.asarray(linear, dtype=np.float32).reshape(3),
                np.asarray(angular, dtype=np.float32).reshape(3),
            )
        policy_dt = POLICY_DECIMATION * 0.005
        if self._prev_position is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        linear_world = (position - self._prev_position) / policy_dt
        angular_world = angular_velocity_from_quaternions(
            self._prev_orientation,
            orientation,
            policy_dt,
        )
        return (
            np.asarray(linear_world, dtype=np.float32).reshape(3),
            np.asarray(angular_world, dtype=np.float32).reshape(3),
        )

    def _infer_policy(self) -> None:
        self.policy_inferences += 1
        position, linear_body, angular_body, gravity_body = self._root_state()
        command = base_vel_cmd()
        self.policy_command = command.copy()
        observation = build_go2_observation(
            linear_body,
            angular_body,
            gravity_body,
            command,
            self.joint_subset.get_joint_positions(),
            self.default_joint_positions,
            self.joint_subset.get_joint_velocities(),
            self.last_action,
            self.observation_dim,
            base_height=float(position[2]),
            ground_height=self.ground_height,
        )
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError('Go2 locomotion policy produced invalid actions')
        self.raw_policy_action = action.astype(np.float32, copy=True)
        self.last_action = self.raw_policy_action.copy()
        self.applied_joint_positions = (
            self.default_joint_positions + DEFAULT_ACTION_SCALE * self.last_action
        )
        holding = getattr(self,'stationary_hold_control',None)
        now = self._simulation_time()
        if holding is not None and now is not None:
            self.applied_joint_positions = holding.update(self.applied_joint_positions,command,
                position,linear_body,angular_body,float(np.degrees(np.arccos(np.clip(-gravity_body[2],-1.,1.)))),
                now,allowed=self.stationary_hold_allowed)
            if holding.state!='tracking':
                # The actor sees the action actually sent to the motor, not
                # an ignored prediction from its continued50Hz inference.
                self.last_action=((self.applied_joint_positions-self.default_joint_positions)/DEFAULT_ACTION_SCALE).astype(np.float32)
