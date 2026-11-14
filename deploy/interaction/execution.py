"""Task-owned identity/deadline; the independent writer enforces each lease."""
import math
from uuid import uuid4


CONTACT_PHASES = frozenset({'engage', 'push', 'settle_push_progress', 'brake_after_push',
    'retreat', 'settle_after_retreat', 'settle_blocked_retreat', 'recover_release_pose',
    'settle_released_corridor', 'verify_passage', 'reobserve'})


class InteractionExecution:
    def __init__(self, profile, duration):
        self.profile, self.duration = profile, float(duration)
        self.reset()

    def reset(self):
        self.identity = self.object_id = None
        self.deadline = self.contact_started = None
        self.segment = 0
        self.awaiting_stop = False

    @property
    def active(self):
        return self.identity is not None

    def start(self, object_id, now):
        self.reset()
        self.identity, self.object_id = uuid4().hex, object_id
        self.deadline = now+self.duration

    def metadata(self, phase, now, source_expires_at):
        if not self.active or self.profile is None:
            return None
        if source_expires_at is None or not math.isfinite(source_expires_at) or source_expires_at <= now:
            raise ValueError('interaction_observation_expired')
        if phase in CONTACT_PHASES and self.contact_started is None:
            self.contact_started = now
            self.deadline = min(self.deadline, now+self.profile.max_contact_seconds)
        if now >= self.deadline:
            raise ValueError('interaction_duration_elapsed')
        p = self.profile
        return {'id': f'{self.identity}:{self.segment}', 'phase': phase,
                'deadline': self.deadline, 'source_expires_at': float(source_expires_at),
                'limits': {'max_vx': max(p.max_forward_mps, p.max_reverse_mps),
                           'max_vy': p.max_lateral_mps, 'max_wz': p.max_yaw_rps,
                           'acceleration': p.linear_acceleration_mps2,
                           'angular_acceleration': p.yaw_acceleration_rps2}}

    def completed(self):
        self.awaiting_stop = self.active

    def stopped(self):
        """Only called on matching worker stopped feedback, never on zero intent."""
        if self.awaiting_stop:
            self.reset()
        elif self.active:
            self.segment += 1  # New writer lease, same absolute task/contact deadline.
