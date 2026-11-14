"""Worker-owned bounds for one real-robot interaction execution.

This envelope bounds commands, not measured force or successful contact.  Only
the motion worker owns this state: planner retries, phase changes and fresh
sensor packets cannot increase an execution's hard deadline or velocity caps.
"""
import math


LIMIT_KEYS = ('max_vx', 'max_vy', 'max_wz', 'acceleration', 'angular_acceleration')


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0.


class ContactGuard:
    def __init__(self, control):
        self.control = control
        self.active = None
        # Keep retired ids across stop/arm. A completed or faulted skill must
        # never resume under an old id with a newly minted duration budget.
        self._used_ids = set()

    def arm(self):
        """Called only after the writer has confirmed stop and accepted arm."""
        self.active = None

    def expiry_reason(self, now):
        if self.active is not None:
            if now >= self.active['deadline']:
                return 'interaction_duration_elapsed'
            if now >= self.active['source_expires_at']:
                return 'interaction_observation_expired'
        return None

    def accept(self, packet, now):
        """Validate an envelope and lock/shrink its immutable execution bounds.

        Raises ValueError carrying a stop reason; callers must latch output off.
        New source samples can renew their own expiry only before the previous
        sample expires. This method cannot repair an already expired lease.
        """
        expired = self.expiry_reason(now)
        if expired:
            raise ValueError(expired)
        if packet is None:
            if self.active is not None:
                raise ValueError('interaction_metadata_missing')
            return
        if not isinstance(packet, dict):
            raise ValueError('invalid_interaction_metadata')
        profile = self.control.get('contact_limits')
        duration = self.control.get('contact_max_duration_s')
        if (not isinstance(profile, dict) or not _positive(duration)
                or any(not _positive(profile.get(key)) for key in LIMIT_KEYS)):
            raise ValueError('interaction_profile_unconfigured')
        identity = packet.get('id')
        phase = packet.get('phase')
        source_expiry = packet.get('source_expires_at')
        deadline = packet.get('deadline')
        limits = packet.get('limits')
        if (not isinstance(identity, str) or not identity.strip() or len(identity) > 256
                or not isinstance(phase, str) or not phase.strip() or len(phase) > 128
                or not _positive(source_expiry) or not _positive(deadline)
                or not isinstance(limits, dict)
                or any(not _positive(limits.get(key)) for key in LIMIT_KEYS)):
            raise ValueError('invalid_interaction_metadata')
        if deadline <= now:
            raise ValueError('interaction_duration_elapsed')
        if source_expiry <= now:
            raise ValueError('interaction_observation_expired')
        if self.active is None:
            if identity in self._used_ids:
                raise ValueError('interaction_id_reused')
            self._used_ids.add(identity)
            self.active = dict(id=identity, phase=phase, source_expires_at=source_expiry,
                               deadline=min(deadline, now+duration), limits={
                                   key: min(self.control[key], profile[key], limits[key])
                                   for key in LIMIT_KEYS})
        else:
            if identity != self.active['id']:
                raise ValueError('interaction_id_changed_without_stop')
            self.active['deadline'] = min(self.active['deadline'], deadline)
            self.active['source_expires_at'] = source_expiry
            self.active['phase'] = phase
            for key in LIMIT_KEYS:
                self.active['limits'][key] = min(self.active['limits'][key], limits[key])

    def effective_config(self):
        if self.active is None:
            return self.control
        return {**self.control, **self.active['limits']}

    def snapshot(self):
        """A serializable diagnostic copy; acknowledgement is not execution."""
        if self.active is None:
            return None
        return {**self.active, 'limits': dict(self.active['limits'])}
