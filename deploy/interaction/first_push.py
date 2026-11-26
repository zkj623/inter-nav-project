"""Select and execute one measured push of a registered box during point navigation."""
import math
from dataclasses import replace

import numpy as np

from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig, PushAsideSkill
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import PushPlan
from deploy.robot.localization import rotation
from .box_motion import ObservedBoxMotion

TRIAL_DISTANCE_M = .10


class RegisteredFacePlanner(SidePushPlanner):
    """Check the registered rectangle, including its fit margin, at the measured yaw."""

    @staticmethod
    def oriented_box(box):
        if box.orientation is None or box.local_dimensions is None:
            return box
        axes = rotation(box.orientation)[:2, :2]
        nominal = np.asarray(box.local_dimensions)
        aabb = abs(axes) @ nominal[:2]
        extra = max(0., float(np.max(np.asarray(box.dimensions[:2])-aabb)))
        dims = (nominal[0]+extra, nominal[1]+extra, nominal[2])
        return replace(box, dimensions=dims, yaw=math.degrees(math.atan2(axes[1, 0], axes[0, 0])),
                       local_dimensions=None, orientation=None)

    def box_mask(self, box, xy=None, padding=0.):
        return super().box_mask(self.oriented_box(box), xy, padding)

    def push_candidates(self, start, goal, boxes, box, optimistic, immovable=()):
        return super().push_candidates(start, goal, boxes, self.oriented_box(box), optimistic, immovable)

    def push_directions(self, box):
        yaw = math.radians(box.yaw)
        u = np.array([math.cos(yaw), math.sin(yaw)])
        v = np.array([-u[1], u[0]])
        return (u, -u, v, -v)

    def first_contact_plan(self, start, boxes, eligible, free, profile):
        """Pick a reachable face with room for a short push and its stopping travel."""
        best = None
        reserve = profile.stopping_distance(profile.push_max_mps, pushing=True,
                                            command_speed=profile.push_max_mps)
        sweep = TRIAL_DISTANCE_M + reserve
        for tracked in boxes:
            if tracked.id not in eligible:
                continue
            box = self.oriented_box(tracked)
            space = free.copy()
            for other in boxes:
                if other.id != box.id:
                    mask = self.box_mask(other, padding=profile.stop_margin_m)
                    if mask is None:
                        space[:] = False
                        break
                    space &= ~mask
            center = np.asarray(box.xy)
            for i, direction in enumerate(self.push_directions(box)):
                axis = i // 2
                dimensions = list(box.dimensions)
                extent = dimensions[axis] / 2
                dimensions[axis] += sweep
                swept_box = replace(box, dimensions=tuple(dimensions))
                mask = self.box_mask(swept_box, xy=center + direction*sweep/2,
                                     padding=profile.stop_margin_m)
                if mask is None or not space[mask].all():
                    continue
                approach = center - direction*(extent + profile.nose_offset_m + PushAsideConfig.front_gap)
                path = self.path(start, approach, boxes)
                if not path:
                    continue
                score = self.length(start, path)
                if best is None or score < best.score:
                    best = PushPlan(box.id, box.xy, tuple(center + direction*TRIAL_DISTANCE_M),
                                    tuple(direction), tuple(approach), path, TRIAL_DISTANCE_M, score)
        return best

    def side_parking_allowed(self, box, destination, direction):
        if box.id not in self.active_passages:
            return False
        axis = np.asarray(self.active_passages[box.id]['axis'])
        # A diagonal face can open the passage if its entire destination
        # rectangle clears the corridor. Keep that clearance and route check.
        if abs(direction @ axis) > .72*np.linalg.norm(axis):
            return False
        return self.clearance(box, destination)['passage_clearance_margin_m'] >= .025

    def clearance(self, box, xy=None):
        if box.id not in self.active_passages:
            # A trial in an open room has no corridor parking objective.
            return {'side_parked': False}
        return super().clearance(self.oriented_box(box), xy)


class FirstPushSkill(PushAsideSkill):
    # Stop-command thresholds. Physical coast is measured separately after stopping.
    stop_displacement = TRIAL_DISTANCE_M
    nose_travel_allowance = .04
    near_seconds = 5.
    stall_seconds = 2.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.near_at = self.stop_at = None
        self.progress_mark_m = 0.
        self.last_progress_at = None
        self.contact_deadline = None
        self.start_base = self.stop_box = self.stop_base = None
        self.maximum_travel = None
        self.review = ObservedBoxMotion(self.backend.profile)
        self.measurement = {}

    def _box_frame(self, ctx):
        # Intersect the nose's line of travel with the real oriented face. The
        # padded world AABB used for route clearance can be several cm closer.
        axes = rotation(ctx.box.orientation)[:2, :2]
        alignment = self.direction @ axes
        axis = int(np.argmax(abs(alignment)))
        normal = axes[:, axis]*np.sign(alignment[axis])
        nose = (ctx.position + rotation(ctx.orientation) @ [self.backend.nose_offset_m, 0., -.06])[:2]
        delta = np.asarray(ctx.box.xy)-nose
        gap = (delta @ normal-ctx.box.local_dimensions[axis]/2)/(self.direction @ normal)
        yaw_error = (self.push_heading-ctx.heading+math.pi) % (2*math.pi)-math.pi
        return float(gap), float(delta @ self.tangent), yaw_error

    def stop_trial(self, now, position, reason, box=None):
        self.stop_at = now
        self.stop_base = np.array(position[:2])
        self.stop_box = None if box is None else np.array(box.xy)
        self.status, self.state, self.reason = 'checking', 'first_push_stop', reason
        return self._output()

    def may_recheck_before_touch(self, ctx):
        if self.near_at is not None or self.engage_box_position is None:
            return False
        delta = np.asarray(ctx.box.xy)-self.engage_box_position
        return self._box_frame(ctx)[0] > .06 and np.linalg.norm(delta) < .015

    def step(self, ctx):
        result = super().step(ctx)
        if (result.status == 'failed' and self.contact_started is not None
                and result.reason in ('sdk_body_or_box_sweep_blocked', 'contact_motion_blocked')
                and ctx.box_observation_id is not None and self.engage_box_position is not None):
            along = (np.asarray(ctx.box.xy) - self.engage_box_position) @ self.direction
            if along >= .02:
                # The next command is blocked. Stop and measure what the push
                # already achieved before declaring its result.
                return self.stop_trial(ctx.now, ctx.position, result.reason, ctx.box)
        return result

    def _step(self, ctx):
        # Preparatory motion keeps the entire box occupied. Starting engagement
        # requires a new scan and a measured stop at the selected face.
        if self.state == 'settle_contact':
            if ctx.box_observation_id is not None:
                reason = self.backend.observe_box(self, ctx)
                if reason:
                    return self._fail(ctx, reason)
            if ctx.box_observation_id is None or not self.backend.box_motion.settled:
                # Body feedback remains available when the scanning beam is
                # elsewhere. Still require a fresh, settled box to leave here.
                self._quiet(ctx, .4)
                return self._output()
        if self.contact_started is not None:
            if self.state not in ('engage', 'push') or ctx.box_observation_id is None:
                return self.stop_trial(ctx.now, ctx.position, 'contact_view_or_alignment_changed')
            gap, _, _ = self._box_frame(ctx)
            if self.start_base is None:
                self.start_base = np.array(ctx.position[:2])
                self.maximum_travel = max(0., gap) + self.nose_travel_allowance
            delta = np.asarray(ctx.box.xy)-self.engage_box_position
            along = float(delta @ self.direction)
            body_travel = np.linalg.norm(ctx.position[:2]-self.start_base)
            if ((self.near_at is None and gap <= .035)
                    or along >= self.progress_mark_m + self.backend.profile.box_position_tolerance_m):
                if self.near_at is None:
                    self.near_at = ctx.now
                self.last_progress_at = ctx.now
                self.progress_mark_m = max(self.progress_mark_m, along)
            reason = None
            if abs(delta @ self.tangent) >= .035 or along < -.02:
                reason = 'box_moved_off_push_line'
            elif along >= self.stop_displacement:
                reason = 'box_reached_trial_distance'
            elif body_travel >= self.maximum_travel + np.clip(along, 0., self.stop_displacement):
                reason = 'first_push_body_travel_limit'
            elif self.near_at is not None and (ctx.now-self.last_progress_at >= self.stall_seconds
                                                or ctx.now-self.near_at >= self.near_seconds-.20):
                reason = 'first_push_time_limit'
            if reason:
                return self.stop_trial(ctx.now, ctx.position, reason, ctx.box)
        result = super()._step(ctx)
        if result.state == 'engage' and self.engage_box_position is not None and self.start_base is None:
            samples = self.backend.box_motion.samples
            if self.backend.box_motion.settled:
                self.engage_box_position = np.mean([corners.mean(axis=0)[:2] for _, corners in samples], axis=0)
        if self.contact_started is not None and self.state not in ('engage', 'push') and result.status == 'running':
            return self.stop_trial(ctx.now, ctx.position, 'contact_phase_finished', ctx.box)
        return result

    def check_stopped_box(self, now, position, box, stamp, scan_id, map_stamp):
        if box is not None and stamp is not None and stamp > self.stop_at:
            self.review.observe(box, stamp, scan_id, now)
            if self.review.settled and map_stamp is not None and stamp <= map_stamp <= now:
                center = np.mean([corners.mean(axis=0)[:2] for _, corners in self.review.samples], axis=0)
                delta = center-self.engage_box_position
                distance, lateral = float(delta @ self.direction), float(delta @ self.tangent)
                self.measurement = dict(displacement_m=distance, lateral_m=lateral,
                    body_coast_m=float(np.linalg.norm(position[:2]-self.stop_base)),
                    box_coast_m=None if self.stop_box is None else float((center-self.stop_box) @ self.direction),
                    stop_reason=self.reason, final_scan_id=str(scan_id), final_scan_at=stamp)
                moved = distance >= .02 and abs(lateral) < .035 and self.reason != 'box_moved_off_push_line'
                self.reason = 'first_push_observed' if moved else 'first_push_displacement_unconfirmed'
                self.status = 'succeeded' if moved else 'failed'
                self.last_box = np.array(box.xy)
                return moved
        if now-self.stop_at >= 8.:
            self.measurement = dict(stop_reason=self.reason)
            self.status, self.reason = 'failed', 'first_push_waiting_for_final_scan'
            return False
        return None

    def report(self):
        return {**super().report(), 'first_push': dict(self.measurement)}
