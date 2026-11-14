"""Recorded stop evidence at the contact-to-navigation boundary.

Replay never has an SDK writer and must not manufacture its ``stopped`` event.
This gate only recognizes consecutive, fresh recorded body-motion samples. The
original provider and map keep updating while task execution waits for evidence.
"""
import math
import numpy as np


class RecordedInteractionHandoff:
    def __init__(self, control, max_feedback_gap_s):
        self.control = control
        self.max_feedback_gap_s = float(max_feedback_gap_s)
        self.requested_at = None
        self.identity = None
        self.stationary_since = self.last_pose_at = None
        self.reason = None

    @property
    def pending(self):
        return self.requested_at is not None

    def _begin(self, tasks, now):
        self.requested_at = float(now)
        self.identity = tasks.engine.execution.identity
        self.stationary_since = self.last_pose_at = None
        self.reason = 'awaiting_fresh_recorded_motion'

    def diagnostic(self):
        return {'pending': self.pending, 'execution_id': self.identity,
                'requested_at': self.requested_at, 'stationary_since': self.stationary_since,
                'last_pose_at': self.last_pose_at, 'reason': self.reason}

    def update(self, tasks, snapshot, now):
        """Return (requested velocity, optional event), without robot output."""
        if not tasks.engine.execution.awaiting_stop:
            velocity = tasks.update(snapshot, now)
            if tasks.engine.execution.awaiting_stop:
                self._begin(tasks, now)
                return None, {'event': 'interaction_stop_handoff_waiting', **self.diagnostic(),
                              'sent_to_robot': False}
            return velocity, None
        if not self.pending:
            self._begin(tasks, now)
        # Do not call active Tasks.update here: it would advance the completed
        # skill's next navigation decision before recorded stopping evidence.
        tasks.observe_only(snapshot, now)
        if not self._recorded_stop(snapshot, now):
            return None, None
        facts = self.diagnostic()
        tasks.engine.interaction_stopped()
        self.requested_at = None
        self.reason = None
        return None, {'event': 'recorded_stop_handoff', **facts, 'pending': False,
                      'recorded_at': float(now), 'evidence_source': 'recorded_body_motion',
                      'sent_to_robot': False}

    def _recorded_stop(self, snapshot, now):
        observation = snapshot.observation
        linear = np.asarray(observation.get('body_linear_velocity', ()), dtype=float)
        angular = np.asarray(observation.get('body_angular_velocity', ()), dtype=float)
        if (linear.shape != (3,) or angular.shape != (3,)
                or not np.isfinite(linear).all() or not np.isfinite(angular).all()):
            self.stationary_since = self.last_pose_at = None
            self.reason = 'complete_recorded_body_motion_missing'
            return False
        pose_at, lidar_at = float(snapshot.pose_at), float(snapshot.lidar_at)
        if (not all(math.isfinite(v) for v in (pose_at, lidar_at, now))
                or min(pose_at, lidar_at) <= self.requested_at
                or not pose_at <= now < pose_at+self.max_feedback_gap_s
                or not lidar_at <= now < lidar_at+self.max_feedback_gap_s):
            self.stationary_since = self.last_pose_at = None
            self.reason = 'awaiting_fresh_recorded_motion'
            return False
        if self.last_pose_at is not None and pose_at <= self.last_pose_at:
            if pose_at < self.last_pose_at:
                self.stationary_since = None
            self.reason = 'awaiting_distinct_recorded_pose'
            return False
        previous_pose = self.last_pose_at
        self.last_pose_at = pose_at
        if (np.linalg.norm(linear) > self.control['stop_speed']
                or np.max(np.abs(angular)) > self.control['stop_yaw_speed']):
            self.stationary_since = None
            self.reason = 'recorded_robot_still_moving'
            return False
        if (self.stationary_since is None or previous_pose is None
                or pose_at-previous_pose > self.max_feedback_gap_s):
            self.stationary_since = pose_at
        self.reason = 'recorded_stop_settling'
        return pose_at-self.stationary_since >= self.control['stop_settle_s']
