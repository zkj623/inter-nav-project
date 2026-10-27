"""Contact preparation uses the same observed-space turn staging as navigation."""
import math
import numpy as np

from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_motion_clear, body_translation_recovery_clear,
)


def checked_contact_turn_command(request, position, heading, planner, boxes, limiter, now):
    """Check both acceleration-limited requests and measured stopping motion.

    All boxes remain obstacles here: deliberate contact starts only after
    preparation. A stationary yaw request is not a stationary physical turn.
    """
    command = limiter.preview(request, now)
    measured = limiter.measured
    horizon = .60 + max(np.linalg.norm(command[:2]), np.linalg.norm(measured[:2])) / (2 * limiter.linear_acceleration)
    safe = body_motion_clear(planner, boxes, position, heading,
        command[0], command[2], horizon=horizon, lateral=command[1])
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
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
        command, safe = checked_contact_turn_command(request, position, heading, planner, boxes, limiter, now)
        return command, {**state, 'controller': 'observed_contact_turn',
            'contact_turn_motion_safe': safe, 'requested_command': request.tolist(),
            'applied_command': command.tolist()}
