"""Pure motion guards; no SDK or ROS dependencies."""
import math
import numpy as np


def checked_velocity(command, config):
    velocity = np.asarray(command, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError('Velocity must be three finite values')
    limits = np.array([config['max_vx'], config['max_vy'], config['max_wz']])
    return velocity / max(1., float(np.max(np.abs(velocity)/limits)))


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

    def arm(self, now):
        self.armed = True
        self.reason = 'active'
        self.expires = now + self.config['command_ttl_s']
        self.velocity.fill(0)
        self.output.fill(0)
        self.last_time = now

    def stop(self, reason):
        self.armed = False
        self.reason = reason
        self.output.fill(0)

    def submit(self, command, generated_at, expires_at, now):
        if not self.armed:
            return False
        if not all(math.isfinite(v) for v in (generated_at, expires_at, now)):
            self.stop('invalid_timestamp')
            return False
        if generated_at > now+.01 or expires_at <= now or expires_at > generated_at+self.config['command_ttl_s']+.001:
            self.stop('expired_command')
            return False
        try:
            self.velocity = checked_velocity(command, self.config)
        except ValueError:
            self.stop('invalid_command')
            return False
        self.expires = expires_at
        return True

    def tick(self, now):
        if not self.armed:
            return None
        if now >= self.expires:
            self.stop('command_timeout')
            return None
        dt = min(max(now-self.last_time, 0), .1)
        delta = np.array([self.config['acceleration']]*2+[self.config['angular_acceleration']])*dt
        self.output += np.clip(self.velocity-self.output, -delta, delta)
        self.last_time = now
        return self.output.copy()
