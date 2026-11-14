"""Pure motion guards; no SDK or ROS dependencies."""
import math
import numpy as np
from .contact_guard import ContactGuard


def checked_velocity(command, config):
    velocity = np.array(command, dtype=float, copy=True)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError('Velocity must be three finite values')
    # The holonomic planner projects a checked map segment into body XY.
    # Preserve that translation direction when either linear axis saturates;
    # yaw saturation must not additionally shrink an otherwise valid stride.
    limits = np.array([config['max_vx'], config['max_vy']])
    velocity[:2] /= max(1., float(np.max(np.abs(velocity[:2])/limits)))
    velocity[2] = np.clip(velocity[2], -config['max_wz'], config['max_wz'])
    return velocity


def check_observation(snapshot, now, robot, control, mapping, camera=False):
    snapshot.require_fresh(now, robot['max_data_age_s'], camera)
    p = snapshot.observation['position']
    w, x, y, z = snapshot.observation['orientation']
    if 1-2*(x*x+y*y) < math.cos(math.radians(control['max_tilt_degrees'])):
        raise ValueError('Excessive body tilt')
    xlim, ylim = mapping['x_limits'], mapping['y_limits']
    margin = mapping['robot_radius']
    if not (xlim[0]+margin < p[0] < xlim[1]-margin and ylim[0]+margin < p[1] < ylim[1]-margin):
        raise ValueError('Robot outside configured map operating boundary')


class CommandGuard:
    """Timeout latches off: later packets cannot silently restart motion."""
    def __init__(self, config):
        self.config = config
        self.armed = False
        self.expires = 0.
        self.velocity = np.zeros(3)
        self.output = np.zeros(3)
        self.reason = 'manual'
        self.last_time = None
        self.deadline = None
        self.contact = ContactGuard(config)

    def arm(self, now, duration_s=None):
        self.contact.arm()
        self.armed = True
        self.reason = 'active'
        self.expires = now + self.config['command_ttl_s']
        self.deadline = None if duration_s is None else now + duration_s
        self.velocity.fill(0)
        self.output.fill(0)
        self.last_time = now

    def stop(self, reason):
        self.armed = False
        self.reason = reason
        self.output.fill(0)

    def check_deadlines(self, now):
        """Latch expired budgets before packets, planning holds or output."""
        if not self.armed:
            return False
        if self.deadline is not None and now >= self.deadline:
            self.stop('test_duration_elapsed')
        elif self.contact.expiry_reason(now):
            self.stop(self.contact.expiry_reason(now))
        elif now >= self.expires:
            self.stop('command_timeout')
        return self.armed

    def submit(self, command, generated_at, expires_at, now, interaction=None):
        if not self.armed:
            return False
        if not all(math.isfinite(v) for v in (generated_at, expires_at, now)):
            self.stop('invalid_timestamp')
            return False
        # The worker receives packets before its next tick. A delayed tick or
        # SDK call must not let a fresh packet renew an already expired lease.
        # Keep the same deadline precedence as tick(), including trial limits.
        if not self.check_deadlines(now):
            return False
        if generated_at > now+.01 or expires_at <= now or expires_at > generated_at+self.config['command_ttl_s']+.001:
            self.stop('expired_command')
            return False
        try:
            self.contact.accept(interaction, now)
        except ValueError as error:
            self.stop(str(error))
            return False
        effective = self.contact.effective_config()
        try:
            self.velocity = checked_velocity(command, effective)
        except ValueError:
            self.stop('invalid_command')
            return False
        # Entering contact or reducing its bounds cannot leave the previous,
        # faster navigation output alive for a deceleration ramp.
        self.output = checked_velocity(self.output, effective)
        self.expires = (expires_at if self.contact.active is None else
                        min(expires_at, self.contact.active['source_expires_at'],
                            self.contact.active['deadline']))
        return True

    def tick(self, now):
        if not self.armed:
            return None
        if not self.check_deadlines(now):
            return None
        dt = min(max(now-self.last_time, 0), .1)
        effective = self.contact.effective_config()
        delta = np.array([effective['acceleration']]*2+[effective['angular_acceleration']])*dt
        self.output += np.clip(self.velocity-self.output, -delta, delta)
        self.output = checked_velocity(self.output, effective)
        self.last_time = now
        return self.output.copy()
