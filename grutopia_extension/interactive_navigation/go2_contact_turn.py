"""Contact preparation uses the same observed-space turn staging as navigation."""
import math
import numpy as np

from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_motion_clear, body_translation_recovery_clear,
)


def checked_contact_turn_command(request, position, heading, planner, boxes, limiter, now, target_position=None):
    """Check both acceleration-limited requests and measured stopping motion.

    All boxes remain obstacles here: deliberate contact starts only after
    preparation. A stationary yaw request is not a stationary physical turn.
    """
    command = limiter.preview(request, now)
    measured = limiter.measured
    horizon = .60 + max(np.linalg.norm(command[:2]), np.linalg.norm(measured[:2])) / (2 * limiter.linear_acceleration)
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
    if target_position is not None:
        delta=np.asarray(target_position[:2])-np.asarray(position[:2])
        world=rotation@command[:2];speed=np.linalg.norm(world);distance=np.linalg.norm(delta)
        if speed>1e-8 and distance>1e-8:
            alignment=float(np.clip(np.dot(world,delta)/(speed*distance),-1.,1.))
            if abs(command[2])<1e-8 and alignment>.95:
                horizon=min(horizon,(distance+.025)/speed)
            elif alignment>.995:
                # Heading hold adds a small yaw to a finite sidestep. Check
                # its complete arc to the goal, not indefinite motion past it.
                # The conservative projected speed covers up to 0.1 rad of
                # turning; larger turns retain the original full horizon.
                cosine=math.cos(math.acos(alignment)+.10)
                finite=(distance+.025)/(speed*cosine)
                if abs(command[2])*finite<=.10:horizon=min(horizon,finite)
            # Actual stopping motion remains independently checked below.
    safe = body_motion_clear(planner, boxes, position, heading,
        command[0], command[2], horizon=horizon, lateral=command[1])
    # Staging may start in a mild existing overlap; allow only withdrawal.
    if not safe and abs(command[2]) < 1e-8:
        target = np.asarray(position[:2]) + rotation @ command[:2] * horizon
        safe = body_translation_recovery_clear(planner, boxes, position, heading,
            target, require_clear_end=False)
    moving = np.linalg.norm(measured[:2]) > .06 or abs(measured[2]) > .10
    if moving:
        momentum_safe = body_motion_clear(planner, boxes, position, heading,
            measured[0], measured[2], horizon=.8, lateral=measured[1])
        if not momentum_safe and abs(command[2]) < 1e-8 and abs(measured[2]) < .1:
            momentum_safe = body_translation_recovery_clear(planner, boxes, position, heading,
                np.asarray(position[:2]) + rotation @ measured[:2] * .8, require_clear_end=False)
        safe = safe and momentum_safe
    return command if safe else np.zeros(3), bool(safe)


def checked_contact_step_command(request, position, heading, planner, boxes, limiter, now, *, target_position):
    """Bound a short correction by its goal, retaining measured stopping space.

    Prefer the original full horizon. The fallback uses the already existing
    finite-goal check, including its independent 0.8 s measured-motion sweep.
    No box, unknown cell, or footprint sample is removed from either check.
    """
    command,safe=checked_contact_turn_command(request,position,heading,planner,boxes,limiter,now)
    if safe:return command,True,'full'
    command,safe=checked_contact_turn_command(request,position,heading,planner,boxes,limiter,now,
                                            target_position=target_position)
    return command,safe,'finite_target' if safe else 'blocked'


class ContactTurnControl:
    def __init__(self, position, heading, goal_heading, planner, boxes, now):
        self.stage = LocalStagingControl(position, heading, goal_heading, planner, boxes, now)
        self.failed = self.stage.failed

    def update(self, position, heading, planner, boxes, limiter, now):
        result = self.stage.update(position, heading, planner, boxes, limiter, now)
        self.failed = self.stage.failed
        if result is None:
            return None
        request, state = result
        target=state['target'] if state['staging_phase']=='staging_move' else None
        command, safe = checked_contact_turn_command(request, position, heading, planner, boxes, limiter, now,target_position=target)
        return command, {**state, 'controller': 'observed_contact_turn',
            'contact_turn_motion_safe': safe, 'requested_command': request.tolist(),
            'applied_command': command.tolist()}
