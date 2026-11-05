"""Isaac-independent state machine for deterministic interaction navigation."""

from dataclasses import dataclass, field
from enum import Enum
from math import atan2, ceil, sqrt
from typing import Any, Optional, Tuple

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Path = Tuple[Vector3, ...]


class InteractionState(str, Enum):
    """High-level phases of the interaction navigation task."""

    NAVIGATE_TO_OBSTACLE = 'navigate_to_obstacle'
    PUSH_OBSTACLE = 'push_obstacle'
    NAVIGATE_TO_OBJECT = 'navigate_to_object'
    REACH_OBJECT = 'reach_object'
    ATTACH_OBJECT = 'attach_object'
    NAVIGATE_TO_CARRY_GOAL = 'navigate_to_carry_goal'
    NAVIGATE_TO_DOOR = 'navigate_to_door'
    REACH_DOOR = 'reach_door'
    PUSH_DOOR = 'push_door'
    NAVIGATE_TO_GOAL = 'navigate_to_goal'
    RECOVER = 'recover'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


class InteractionEffect(str, Enum):
    """Simulator-side effects requested by the pure state machine."""

    NONE = 'none'
    LOCK_ROBOT = 'lock_robot'
    UNLOCK_ROBOT = 'unlock_robot'
    ENABLE_OBJECT_COLLISION = 'enable_object_collision'
    ATTACH_OBJECT = 'attach_object'


class FailureReason(str, Enum):
    """Terminal failure categories exposed to demos and tests."""

    STATE_TIMEOUT = 'state_timeout'
    RECOVERY_LIMIT = 'recovery_limit'
    ATTACHMENT_LOST = 'attachment_lost'
    STEP_WENT_BACKWARDS = 'step_went_backwards'


@dataclass(frozen=True)
class ControllerNames:
    """GRUtopia controller names used by the state machine."""

    path: str = 'move_along_path'
    speed: str = 'move_by_speed'
    arm_ik: str = 'right_arm_ik_controller'
    recover: str = 'recover'


@dataclass(frozen=True)
class ControllerCommand:
    """One controller invocation in GRUtopia's action-dict format."""

    name: str
    data: Tuple[Any, ...]

    def as_action(self) -> dict:
        return {self.name: list(self.data)}


@dataclass(frozen=True)
class InteractionPlan:
    """Geometry and waypoints for one deterministic task instance."""

    obstacle_path: Path
    object_path: Path
    door_path: Path
    goal_path: Path
    pick_offset: Vector3
    door_contact_position: Vector3
    door_push_position: Vector3
    arm_orientation: Optional[Quaternion] = None
    carry_path: Optional[Path] = None

    def __post_init__(self):
        for name in ('obstacle_path', 'object_path', 'door_path', 'goal_path'):
            path = getattr(self, name)
            if not path:
                raise ValueError(f'{name} must contain at least one waypoint')
            for point in path:
                _validate_vector(point, 3, f'{name} waypoint')
        _validate_vector(self.pick_offset, 3, 'pick_offset')
        _validate_vector(self.door_contact_position, 3, 'door_contact_position')
        _validate_vector(self.door_push_position, 3, 'door_push_position')
        if self.arm_orientation is not None:
            _validate_vector(self.arm_orientation, 4, 'arm_orientation')
        if self.carry_path is not None:
            if not self.carry_path:
                raise ValueError('carry_path must contain at least one waypoint')
            for point in self.carry_path:
                _validate_vector(point, 3, 'carry_path waypoint')


@dataclass(frozen=True)
class StateTimeouts:
    """Maximum simulation steps allowed in each non-terminal phase."""

    navigate_to_obstacle: int = 3000
    push_obstacle: int = 1200
    navigate_to_object: int = 2400
    reach_object: int = 1200
    attach_object: int = 240
    navigate_to_carry_goal: int = 3600
    navigate_to_door: int = 2400
    reach_door: int = 1200
    push_door: int = 3200
    navigate_to_goal: int = 3600
    recover: int = 480

    def for_state(self, state: InteractionState) -> Optional[int]:
        if state in (InteractionState.SUCCEEDED, InteractionState.FAILED):
            return None
        return getattr(self, state.value)

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class StateMachineConfig:
    """Thresholds and control values that are independent of scene geometry."""

    navigation_tolerance: float = 0.15
    minimum_object_navigation_distance: float = 0.0
    door_navigation_tolerance: float = 0.20
    door_contact_angle: float = 0.008
    manipulation_heading_tolerance: float = 0.20
    hand_tolerance: float = 0.19
    door_hand_tolerance: float = 0.26
    ik_waypoint_max_step: float = 10.0
    ik_waypoint_tolerance: float = 0.04
    object_ik_spatial_fallback_steps: int = 0
    object_heading: float = 0.0
    object_navigation_settle_tolerance: float = 0.0
    object_navigation_settle_steps: int = 0
    require_object_hand_contact: bool = False
    obstacle_displacement: float = 0.30
    door_open_angle: float = 0.02
    push_forward_speed: float = 0.45
    door_base_push_speed: float = 0.20
    fall_height: float = 0.55
    recovered_height: float = 0.90
    recover_height: float = 1.05
    recover_settle_steps: int = 12
    max_recoveries: int = 8
    enable_recovery: bool = True
    timeouts: StateTimeouts = field(default_factory=StateTimeouts)

    def __post_init__(self):
        positive_values = {
            'navigation_tolerance': self.navigation_tolerance,
            'door_navigation_tolerance': self.door_navigation_tolerance,
            'door_contact_angle': self.door_contact_angle,
            'manipulation_heading_tolerance': self.manipulation_heading_tolerance,
            'hand_tolerance': self.hand_tolerance,
            'door_hand_tolerance': self.door_hand_tolerance,
            'ik_waypoint_max_step': self.ik_waypoint_max_step,
            'ik_waypoint_tolerance': self.ik_waypoint_tolerance,
            'obstacle_displacement': self.obstacle_displacement,
            'door_open_angle': self.door_open_angle,
            'push_forward_speed': self.push_forward_speed,
            'door_base_push_speed': self.door_base_push_speed,
            'fall_height': self.fall_height,
            'recovered_height': self.recovered_height,
            'recover_height': self.recover_height,
            'recover_settle_steps': self.recover_settle_steps,
            'max_recoveries': self.max_recoveries,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f'{name} must be positive')
        if self.minimum_object_navigation_distance < 0:
            raise ValueError('minimum_object_navigation_distance cannot be negative')
        if self.object_ik_spatial_fallback_steps < 0:
            raise ValueError('object_ik_spatial_fallback_steps cannot be negative')
        if self.object_navigation_settle_tolerance < 0:
            raise ValueError('object_navigation_settle_tolerance cannot be negative')
        if self.object_navigation_settle_steps < 0:
            raise ValueError('object_navigation_settle_steps cannot be negative')
        if self.fall_height >= self.recovered_height:
            raise ValueError('fall_height must be smaller than recovered_height')


@dataclass(frozen=True)
class InteractionObservation:
    """Minimal simulator observation consumed by the state machine."""

    step: int
    robot_position: Vector3
    hand_position: Vector3
    obstacle_position: Vector3
    object_position: Vector3
    door_open_angle: float
    object_attached: bool = False
    robot_yaw: float = 0.0
    ik_success: Optional[bool] = None
    ik_finished: Optional[bool] = None
    ik_position_error: Optional[float] = None
    door_arm_contact: bool = False
    object_hand_contact: bool = False

    def __post_init__(self):
        if self.step < 0:
            raise ValueError('step must be non-negative')
        _validate_vector(self.robot_position, 3, 'robot_position')
        _validate_vector(self.hand_position, 3, 'hand_position')
        _validate_vector(self.obstacle_position, 3, 'obstacle_position')
        _validate_vector(self.object_position, 3, 'object_position')


@dataclass(frozen=True)
class StateMachineDecision:
    """Action, side effect, and status produced for one simulation step."""

    state: InteractionState
    command: Optional[ControllerCommand] = None
    support_commands: Tuple[ControllerCommand, ...] = ()
    effect: InteractionEffect = InteractionEffect.NONE
    success: bool = False
    failure_reason: Optional[FailureReason] = None
    message: str = ''

    @property
    def terminal(self) -> bool:
        return self.state in (InteractionState.SUCCEEDED, InteractionState.FAILED)

    def as_action(self) -> dict:
        action = {}
        for support_command in self.support_commands:
            action.update(support_command.as_action())
        if self.command is not None:
            action.update(self.command.as_action())
        return action


class InteractionNavigationStateMachine:
    """Deterministic, simulation-step-driven interaction controller."""

    _POST_ATTACH_STATES = {
        InteractionState.NAVIGATE_TO_CARRY_GOAL,
        InteractionState.NAVIGATE_TO_DOOR,
        InteractionState.REACH_DOOR,
        InteractionState.PUSH_DOOR,
        InteractionState.NAVIGATE_TO_GOAL,
    }

    def __init__(
        self,
        plan: InteractionPlan,
        config: StateMachineConfig = StateMachineConfig(),
        controllers: ControllerNames = ControllerNames(),
        initial_state: InteractionState = InteractionState.NAVIGATE_TO_OBSTACLE,
    ):
        if initial_state in (InteractionState.RECOVER, InteractionState.SUCCEEDED, InteractionState.FAILED):
            raise ValueError('initial_state must be an active task phase')
        self.plan = plan
        self.config = config
        self.controllers = controllers
        self.initial_state = initial_state
        self.state = initial_state
        self.failure_reason: Optional[FailureReason] = None
        self._entered_step: Optional[int] = None
        self._last_step: Optional[int] = None
        self._obstacle_start: Optional[Vector3] = None
        self._resume_state: Optional[InteractionState] = None
        self._attach_effect_sent = False
        self._pending_effect = InteractionEffect.NONE
        self._recoveries = {}
        self._object_ik_waypoints: Path = ()
        self._object_ik_waypoint_index = 0
        self._object_ik_spatial_stable_steps = 0
        self._object_ik_convergence_mode: Optional[str] = None
        self._door_ik_waypoints: Path = ()
        self._door_ik_waypoint_index = 0
        self._object_navigation_origin: Optional[Vector3] = None
        self._object_navigation_distance = 0.0
        self._object_navigation_settle_steps = 0
        self._object_navigation_lock_mode: Optional[str] = None

    @property
    def object_navigation_distance(self) -> float:
        return self._object_navigation_distance

    @property
    def object_navigation_lock_mode(self) -> Optional[str]:
        return self._object_navigation_lock_mode

    @property
    def object_ik_convergence_mode(self) -> Optional[str]:
        return self._object_ik_convergence_mode

    def step(self, observation: InteractionObservation) -> StateMachineDecision:
        """Advance at most one phase and return the command for the resulting state."""

        if self._last_step is not None and observation.step < self._last_step:
            return self._fail(FailureReason.STEP_WENT_BACKWARDS, 'observation step went backwards')
        self._last_step = observation.step
        if self._entered_step is None:
            self._entered_step = observation.step

        if self.state in (InteractionState.SUCCEEDED, InteractionState.FAILED):
            return self._decision()

        if self.state != InteractionState.RECOVER and observation.robot_position[2] < self.config.fall_height:
            if not self.config.enable_recovery:
                return self._fail(FailureReason.RECOVERY_LIMIT, 'recovery is disabled for this robot')
            recoveries = self._recoveries.get(self.state, 0)
            if recoveries >= self.config.max_recoveries:
                return self._fail(FailureReason.RECOVERY_LIMIT, 'maximum recovery attempts exceeded')
            self._recoveries[self.state] = recoveries + 1
            self._resume_state = self.state
            self._enter(InteractionState.RECOVER, observation)
            return self._decision(observation)

        if self.state == InteractionState.RECOVER:
            if (
                observation.robot_position[2] >= self.config.recovered_height
                and self._elapsed(observation) >= self.config.recover_settle_steps
            ):
                resume_state = self._resume_state or self.initial_state
                self._resume_state = None
                self._enter(resume_state, observation)
            elif self._timed_out(observation):
                return self._fail(FailureReason.STATE_TIMEOUT, 'recover timed out')
            return self._decision(observation)

        if self.state in self._POST_ATTACH_STATES and not observation.object_attached:
            return self._fail(FailureReason.ATTACHMENT_LOST, 'carried object is no longer attached')

        self._advance(observation)
        if self.state not in (InteractionState.SUCCEEDED, InteractionState.FAILED) and self._timed_out(observation):
            return self._fail(FailureReason.STATE_TIMEOUT, f'{self.state.value} timed out')
        return self._decision(observation)

    def _advance(self, observation: InteractionObservation):
        if self.state == InteractionState.NAVIGATE_TO_OBSTACLE:
            if _distance_xy(observation.robot_position, self.plan.obstacle_path[-1]) <= self.config.navigation_tolerance:
                self._enter(InteractionState.PUSH_OBSTACLE, observation)
            return

        if self.state == InteractionState.PUSH_OBSTACLE:
            if self._obstacle_start is None:
                self._obstacle_start = observation.obstacle_position
            if _distance_xy(observation.obstacle_position, self._obstacle_start) >= self.config.obstacle_displacement:
                self._enter(InteractionState.NAVIGATE_TO_OBJECT, observation)
            return

        if self.state == InteractionState.NAVIGATE_TO_OBJECT:
            if self._object_navigation_origin is None:
                self._object_navigation_origin = observation.robot_position
            self._object_navigation_distance = max(
                self._object_navigation_distance,
                _distance_xy(observation.robot_position, self._object_navigation_origin),
            )
            lock_mode = self._object_navigation_lock_mode_for(observation)
            if lock_mode is not None:
                self._object_navigation_lock_mode = lock_mode
                self._enter(InteractionState.REACH_OBJECT, observation)
            return

        if self.state == InteractionState.REACH_OBJECT:
            pick_position = _add(observation.object_position, self.plan.pick_offset)
            if (
                self._object_ik_waypoints
                and _distance(self._object_ik_waypoints[-1], pick_position)
                > self.config.ik_waypoint_tolerance
            ):
                self._set_object_ik_waypoints(observation)
            waypoint = self._object_ik_target(pick_position)
            if self._object_ik_target_reached(observation, waypoint):
                if self._object_ik_waypoint_index + 1 < len(self._object_ik_waypoints):
                    self._object_ik_waypoint_index += 1
                    self._object_ik_spatial_stable_steps = 0
                    self._object_ik_convergence_mode = None
                elif _distance(observation.hand_position, pick_position) <= self.config.hand_tolerance:
                    if (
                        not self.config.require_object_hand_contact
                        or observation.object_hand_contact
                    ):
                        self._enter(InteractionState.ATTACH_OBJECT, observation)
            elif (
                self.config.require_object_hand_contact
                and observation.object_hand_contact
                and _distance(observation.hand_position, pick_position) <= self.config.hand_tolerance
            ):
                self._object_ik_convergence_mode = (
                    self._object_ik_convergence_mode or 'contact_fallback'
                )
                self._enter(InteractionState.ATTACH_OBJECT, observation)
            return

        if self.state == InteractionState.ATTACH_OBJECT:
            if observation.object_attached:
                self._enter(
                    (
                        InteractionState.NAVIGATE_TO_CARRY_GOAL
                        if self.plan.carry_path is not None
                        else InteractionState.NAVIGATE_TO_DOOR
                    ),
                    observation,
                )
            return

        if self.state == InteractionState.NAVIGATE_TO_CARRY_GOAL:
            if (
                self.plan.carry_path is not None
                and _distance_xy(observation.robot_position, self.plan.carry_path[-1])
                <= self.config.navigation_tolerance
            ):
                self._enter(InteractionState.SUCCEEDED, observation)
            return

        if self.state == InteractionState.NAVIGATE_TO_DOOR:
            if (
                (
                    _distance_xy(observation.robot_position, self.plan.door_path[-1])
                    <= self.config.door_navigation_tolerance
                    and abs(
                        _normalized_angle(
                            observation.robot_yaw - self._door_push_heading()
                        )
                    )
                    <= self.config.manipulation_heading_tolerance
                )
                or abs(observation.door_open_angle) >= self.config.door_contact_angle
            ):
                self._enter(InteractionState.REACH_DOOR, observation)
            return

        if self.state == InteractionState.REACH_DOOR:
            waypoint = self._door_ik_target()
            if (
                _distance(observation.hand_position, waypoint) <= self.config.ik_waypoint_tolerance
                and (_ik_converged(observation) or observation.door_arm_contact)
            ):
                if self._door_ik_waypoint_index + 1 < len(self._door_ik_waypoints):
                    self._door_ik_waypoint_index += 1
                elif (
                    _distance(observation.hand_position, self.plan.door_contact_position)
                    <= self.config.door_hand_tolerance
                ):
                    self._enter(InteractionState.PUSH_DOOR, observation)
            return

        if self.state == InteractionState.PUSH_DOOR:
            if abs(observation.door_open_angle) >= self.config.door_open_angle:
                self._enter(InteractionState.NAVIGATE_TO_GOAL, observation)
            return

        if self.state == InteractionState.NAVIGATE_TO_GOAL:
            if _distance_xy(observation.robot_position, self.plan.goal_path[-1]) <= self.config.navigation_tolerance:
                self._enter(InteractionState.SUCCEEDED, observation)

    def _enter(self, state: InteractionState, observation: InteractionObservation):
        self.state = state
        self._entered_step = observation.step
        self._pending_effect = {
            InteractionState.REACH_OBJECT: InteractionEffect.LOCK_ROBOT,
            InteractionState.NAVIGATE_TO_CARRY_GOAL: InteractionEffect.UNLOCK_ROBOT,
            InteractionState.NAVIGATE_TO_DOOR: InteractionEffect.UNLOCK_ROBOT,
            InteractionState.REACH_DOOR: InteractionEffect.LOCK_ROBOT,
            InteractionState.PUSH_DOOR: InteractionEffect.ENABLE_OBJECT_COLLISION,
            InteractionState.NAVIGATE_TO_GOAL: InteractionEffect.UNLOCK_ROBOT,
            InteractionState.RECOVER: InteractionEffect.UNLOCK_ROBOT,
        }.get(state, InteractionEffect.NONE)
        if state == InteractionState.PUSH_OBSTACLE:
            self._obstacle_start = observation.obstacle_position
        if state == InteractionState.NAVIGATE_TO_OBJECT:
            self._object_navigation_origin = observation.robot_position
            self._object_navigation_distance = 0.0
            self._object_navigation_settle_steps = 0
            self._object_navigation_lock_mode = None
        if state == InteractionState.REACH_OBJECT:
            self._set_object_ik_waypoints(observation)
        if state == InteractionState.REACH_DOOR:
            self._set_door_ik_waypoints(observation)
        if state == InteractionState.ATTACH_OBJECT:
            self._attach_effect_sent = False

    def _decision(self, observation: Optional[InteractionObservation] = None) -> StateMachineDecision:
        if self.state == InteractionState.SUCCEEDED:
            return StateMachineDecision(state=self.state, success=True, message='goal reached with carried object')
        if self.state == InteractionState.FAILED:
            return StateMachineDecision(
                state=self.state,
                failure_reason=self.failure_reason,
                message=self.failure_reason.value if self.failure_reason is not None else 'failed',
            )
        if observation is None:
            return StateMachineDecision(state=self.state)

        command: Optional[ControllerCommand] = None
        support_commands: Tuple[ControllerCommand, ...] = ()
        effect = self._pending_effect
        self._pending_effect = InteractionEffect.NONE
        if self.state == InteractionState.NAVIGATE_TO_OBSTACLE:
            command = self._path_command(self.plan.obstacle_path)
        elif self.state == InteractionState.PUSH_OBSTACLE:
            command = ControllerCommand(
                self.controllers.speed,
                (self.config.push_forward_speed, 0.0, 0.0),
            )
        elif self.state == InteractionState.NAVIGATE_TO_OBJECT:
            command = self._path_command(self.plan.object_path)
        elif self.state == InteractionState.REACH_OBJECT:
            support_commands = (self._balance_command(),)
            pick_position = _add(observation.object_position, self.plan.pick_offset)
            command = self._ik_command(self._object_ik_target(pick_position))
        elif self.state == InteractionState.ATTACH_OBJECT:
            support_commands = (self._balance_command(),)
            command = ControllerCommand(self.controllers.arm_ik, (None, None))
            if not self._attach_effect_sent:
                effect = InteractionEffect.ATTACH_OBJECT
                self._attach_effect_sent = True
        elif self.state == InteractionState.NAVIGATE_TO_CARRY_GOAL:
            command = self._path_command(self.plan.carry_path or self.plan.goal_path)
        elif self.state == InteractionState.NAVIGATE_TO_DOOR:
            command = self._path_command(self.plan.door_path)
        elif self.state == InteractionState.REACH_DOOR:
            support_commands = (self._balance_command(),)
            command = self._ik_command(self._door_ik_target())
        elif self.state == InteractionState.PUSH_DOOR:
            support_commands = (
                ControllerCommand(self.controllers.speed, (self.config.door_base_push_speed, 0.0, 0.0)),
            )
            command = self._ik_command(self.plan.door_push_position)
        elif self.state == InteractionState.NAVIGATE_TO_GOAL:
            command = self._path_command(self.plan.goal_path)
        elif self.state == InteractionState.RECOVER:
            recover_xy = self._recover_xy(observation)
            recover_position = (
                recover_xy[0],
                recover_xy[1],
                self.config.recover_height,
            )
            command = ControllerCommand(self.controllers.recover, (recover_position, (1.0, 0.0, 0.0, 0.0)))
        return StateMachineDecision(
            state=self.state,
            command=command,
            support_commands=support_commands,
            effect=effect,
        )

    def _path_command(self, path: Path) -> ControllerCommand:
        return ControllerCommand(self.controllers.path, (path,))

    def _ik_command(self, position: Vector3) -> ControllerCommand:
        return ControllerCommand(
            self.controllers.arm_ik,
            (position, self.plan.arm_orientation, self.plan.arm_orientation is None),
        )

    def _balance_command(self) -> ControllerCommand:
        return ControllerCommand(self.controllers.speed, (0.0, 0.0, 0.0))

    def _set_object_ik_waypoints(self, observation: InteractionObservation):
        pick_position = _add(observation.object_position, self.plan.pick_offset)
        self._object_ik_waypoints = _linear_waypoints(
            observation.hand_position,
            pick_position,
            self.config.ik_waypoint_max_step,
        )
        self._object_ik_waypoint_index = 0
        self._object_ik_spatial_stable_steps = 0
        self._object_ik_convergence_mode = None

    def _object_navigation_lock_mode_for(self, observation: InteractionObservation) -> Optional[str]:
        distance = _distance_xy(observation.robot_position, self.plan.object_path[-1])
        heading_error = abs(
            _normalized_angle(observation.robot_yaw - self.config.object_heading)
        )
        heading_ok = heading_error <= self.config.manipulation_heading_tolerance
        progress_ok = (
            self._object_navigation_distance >= self.config.minimum_object_navigation_distance
        )
        if (
            distance <= self.config.navigation_tolerance
            and heading_ok
            and progress_ok
        ):
            self._object_navigation_settle_steps = 0
            return 'precise'
        settle_tolerance = self.config.object_navigation_settle_tolerance
        settle_steps = self.config.object_navigation_settle_steps
        if (
            settle_tolerance > 0
            and settle_steps > 0
            and distance <= settle_tolerance
            and heading_ok
            and progress_ok
        ):
            self._object_navigation_settle_steps += 1
            if self._object_navigation_settle_steps >= settle_steps:
                return 'settle_fallback'
            return None
        self._object_navigation_settle_steps = 0
        return None

    def _object_ik_target_reached(
        self,
        observation: InteractionObservation,
        waypoint: Vector3,
    ) -> bool:
        tolerance = self.config.ik_waypoint_tolerance
        if _distance(observation.hand_position, waypoint) > tolerance:
            self._object_ik_spatial_stable_steps = 0
            self._object_ik_convergence_mode = None
            return False
        if _ik_converged(observation):
            self._object_ik_spatial_stable_steps = 0
            self._object_ik_convergence_mode = 'solver'
            return True
        fallback_steps = self.config.object_ik_spatial_fallback_steps
        if (
            fallback_steps <= 0
            or observation.ik_position_error is None
            or observation.ik_position_error > tolerance
        ):
            self._object_ik_spatial_stable_steps = 0
            self._object_ik_convergence_mode = None
            return False
        self._object_ik_spatial_stable_steps += 1
        if self._object_ik_spatial_stable_steps < fallback_steps:
            return False
        self._object_ik_convergence_mode = 'spatial_fallback'
        return True

    def _object_ik_target(self, fallback: Vector3) -> Vector3:
        if not self._object_ik_waypoints:
            return fallback
        index = min(self._object_ik_waypoint_index, len(self._object_ik_waypoints) - 1)
        return self._object_ik_waypoints[index]

    def _set_door_ik_waypoints(self, observation: InteractionObservation):
        self._door_ik_waypoints = _linear_waypoints(
            observation.hand_position,
            self.plan.door_contact_position,
            self.config.ik_waypoint_max_step,
        )
        self._door_ik_waypoint_index = 0

    def _door_ik_target(self) -> Vector3:
        if not self._door_ik_waypoints:
            return self.plan.door_contact_position
        index = min(self._door_ik_waypoint_index, len(self._door_ik_waypoints) - 1)
        return self._door_ik_waypoints[index]

    def _door_push_heading(self) -> float:
        delta_x = self.plan.door_push_position[0] - self.plan.door_contact_position[0]
        delta_y = self.plan.door_push_position[1] - self.plan.door_contact_position[1]
        return atan2(delta_y, delta_x)

    def _recover_xy(self, observation: InteractionObservation) -> Tuple[float, float]:
        if self._resume_state == InteractionState.NAVIGATE_TO_OBJECT:
            return self.plan.obstacle_path[-1][:2]
        if self._resume_state in (InteractionState.REACH_OBJECT, InteractionState.ATTACH_OBJECT):
            return self.plan.object_path[-1][:2]
        if self._resume_state == InteractionState.NAVIGATE_TO_DOOR:
            return self.plan.object_path[-1][:2]
        if self._resume_state == InteractionState.NAVIGATE_TO_CARRY_GOAL:
            return self.plan.object_path[-1][:2]
        if self._resume_state in (InteractionState.REACH_DOOR, InteractionState.PUSH_DOOR):
            return self.plan.door_path[-1][:2]
        if self._resume_state == InteractionState.NAVIGATE_TO_GOAL:
            return self.plan.door_path[-1][:2]
        return observation.robot_position[:2]

    def _elapsed(self, observation: InteractionObservation) -> int:
        return observation.step - (self._entered_step if self._entered_step is not None else observation.step)

    def _timed_out(self, observation: InteractionObservation) -> bool:
        timeout = self.config.timeouts.for_state(self.state)
        return timeout is not None and self._elapsed(observation) >= timeout

    def _fail(self, reason: FailureReason, message: str) -> StateMachineDecision:
        self.state = InteractionState.FAILED
        self.failure_reason = reason
        return StateMachineDecision(state=self.state, failure_reason=reason, message=message)


def _validate_vector(value: tuple, length: int, name: str):
    if len(value) != length:
        raise ValueError(f'{name} must contain {length} values')


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[idx] + right[idx] for idx in range(3))


def _distance(left: Vector3, right: Vector3) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(3)))


def _distance_xy(left: Vector3, right: Vector3) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(2)))


def _linear_waypoints(start: Vector3, target: Vector3, maximum_step: float) -> Path:
    distance = _distance(start, target)
    count = max(1, int(ceil(distance / maximum_step)))
    return tuple(
        tuple(start[axis] + (target[axis] - start[axis]) * index / count for axis in range(3))
        for index in range(1, count + 1)
    )


def _ik_converged(observation: InteractionObservation) -> bool:
    """Treat missing legacy IK status as unknown, but honor explicit failures."""

    return observation.ik_success is not False and observation.ik_finished is not False


def _normalized_angle(angle: float) -> float:
    from math import pi

    return (angle + pi) % (2.0 * pi) - pi
