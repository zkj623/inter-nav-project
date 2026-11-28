"""Select and execute one measured push of a registered box during point navigation."""
from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass, replace

import numpy as np

from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig, PushAsideSkill
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState, PushPlan
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear, body_translation_recovery_clear
from deploy.robot.localization import rotation
from .box_motion import ObservedBoxMotion
from .collision import box_sweep_space
from .motion import SdkNavigationMotion

TRIAL_DISTANCE_M = .10
SHORT_END_PUSH_DISTANCE_M = .40 * 4 / 3
SHORT_END_APPROACH_MAX_MPS = .35
# Run f407ca52 averaged 0.09 m/s while covering the initial gap at a 0.20 m/s request.
SHORT_END_APPROACH_BUDGET_MPS = .08
TRIAL_YAW_TOLERANCE = math.radians(15.)
# The field robot can spring back to 6.7° as it settles. Keep the tighter
# aiming band during a turn and use the settled band through lid centering.
SHORT_END_YAW_TOLERANCE = math.radians(6.)
SHORT_END_SETTLED_YAW_TOLERANCE = math.radians(7.)
SIDE_STAGE_ARRIVAL_M = .08
NEARBY_TURN_RADIUS_M = .16
MAX_SIDE_STAGE_OFFSET_M = .32
SHORT_END_MAX_STANDOFF_M = .75
SHORT_END_CENTER_BAND_M = .05
SHORT_END_LANDING_OFFSET_M = .08


@dataclass
class TimedFacePush:
    box: BoxState
    origin: np.ndarray
    gap: float
    distance: float
    speed: float
    planned_at: float
    scan_max_age: float
    forward: np.ndarray
    face_reached_at: float | None = None
    sdk_started_at: float | None = None

    @property
    def approach_speed(self):
        return min(self.speed, SHORT_END_APPROACH_MAX_MPS)

    @property
    def approach_timeout(self):
        return self.gap / min(SHORT_END_APPROACH_BUDGET_MPS, self.approach_speed)

    @property
    def stroke_seconds(self):
        # Preserve the pre-increase stroke budget on a slowing, loaded gait.
        # The odometry distance limit can finish a faster stroke sooner.
        return self.distance / self.approach_speed

    @property
    def duration(self):
        approach = self.approach_timeout
        if self.face_reached_at is not None:
            approach = min(approach, self.face_reached_at - self.sdk_started_at)
        return approach + self.stroke_seconds


def face_turn_profile(profile, navigation_yaw_limit):
    # At the stand-off pose the field robot barely turned at 0.32 rad/s.
    # Navigation turns in the same run made progress at about 0.5–0.6 rad/s.
    cap = min(max(.60, profile.max_yaw_rps), navigation_yaw_limit)
    # Include the angle swept while the faster request ramps down to zero.
    stop_seconds = profile.yaw_stop_seconds + cap / (2*profile.yaw_acceleration_rps2)
    return replace(profile, min_yaw_rps=min(.50, cap), max_yaw_rps=cap, yaw_stop_seconds=stop_seconds)


def trial_sweep_clear(planner, tracked, direction, boxes, free, profile, push_distance=TRIAL_DISTANCE_M,
                      *, speed=None):
    """Check just this face on the current map, including a short push and braking."""
    box = RegisteredFacePlanner.oriented_box(tracked)
    direction = np.asarray(direction)
    speed = profile.push_max_mps if speed is None else speed
    distance = push_distance + profile.stopping_distance(speed, pushing=True, command_speed=speed)
    # The measured box can turn a few degrees within the fit uncertainty
    # while the chosen world direction stays fixed. Cover both local axes.
    yaw = math.radians(box.yaw)
    axes = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    size = np.asarray(box.dimensions[:2]) + distance * abs(direction @ axes)
    swept = replace(box, dimensions=(*size, box.dimensions[2]))
    mask = planner.box_mask(swept, xy=np.asarray(box.xy) + direction * distance/2,
                            padding=profile.stop_margin_m)
    if mask is None or not box_sweep_space(planner, free)[mask].all():
        return False
    mixed = getattr(planner, 'mixed_box_cells', {}).get(box.id)
    if mixed is not None and (mask & mixed).any():
        return False
    for other in boxes:
        if other.id != box.id:
            obstacle = planner.box_mask(other, padding=profile.stop_margin_m)
            if obstacle is None or (mask & obstacle).any():
                return False
    return True


def trial_speeds(profile):
    fast, start = profile.push_max_mps, profile.push_start_mps
    return dict.fromkeys((fast, (fast + start) / 2, start))


def trial_push_speed(planner, box, direction, boxes, free, profile, push_distance=TRIAL_DISTANCE_M):
    """Select a contact speed whose full push and braking reserve fit the observed map."""
    for speed in trial_speeds(profile):
        if trial_sweep_clear(planner, box, direction, boxes, free, profile, push_distance, speed=speed):
            return speed
    return None


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
        mask = super().box_mask(self.oriented_box(box), xy, padding)
        surface = getattr(self, 'measured_box_cells', {}).get(box.id)
        if mask is not None and xy is None and surface is not None and np.array_equal(box.xy, surface[0]):
            mask |= surface[1]
        return mask

    def push_candidates(self, start, goal, boxes, box, optimistic, immovable=()):
        return super().push_candidates(start, goal, boxes, self.oriented_box(box), optimistic, immovable)

    def push_directions(self, box):
        yaw = math.radians(box.yaw)
        u = np.array([math.cos(yaw), math.sin(yaw)])
        v = np.array([-u[1], u[0]])
        return (u, -u, v, -v)

    @staticmethod
    def _centering_path_clear(planner, boxes, stage, heading, tangent, lateral, face_width, profile):
        # Stop at the first usable part of the face. On the 30 cm short end
        # this leaves 5 cm between the head edge and the box edge at zero yaw.
        band = np.clip(face_width / 2 - profile.head_half_width_m - .05, 0., SHORT_END_CENTER_BAND_M)
        travel = lateral - np.clip(lateral, -band, band)
        return body_translation_recovery_clear(planner, boxes, stage, heading, stage + travel*tangent)

    def nearby_side_stage_clear(self, position, heading, plan, boxes, profile):
        """Check turning and centering from the actual stopped side-stage pose."""
        xy = np.asarray(position[:2])
        direction = np.asarray(plan.direction)
        tangent = np.array([-direction[1], direction[0]])
        box = next((b for b in boxes if b.id == plan.box_id), None)
        if box is None or box.local_dimensions is None or box.orientation is None:
            return False
        delta = np.asarray(box.xy) - xy
        lateral = float(delta @ tangent)
        axes = rotation(box.orientation)[:2, :2]
        half_extent = abs(direction @ axes) @ np.asarray(box.local_dimensions[:2]) / 2
        nose_gap = delta @ direction - half_extent - profile.nose_offset_m
        if abs(lateral) > MAX_SIDE_STAGE_OFFSET_M or not .20 <= nose_gap <= SHORT_END_MAX_STANDOFF_M:
            return False
        check = copy(self)
        SdkNavigationMotion(profile).bind(check)
        target_yaw = math.atan2(direction[1], direction[0])
        end = heading + (target_yaw - heading + math.pi) % (2 * math.pi) - math.pi
        return (body_rotation_clear(check, boxes, xy, heading, end)
                and self._centering_path_clear(check, boxes, xy, end, tangent, lateral,
                                               min(box.local_dimensions[:2]), profile))

    def turn_standoff(self, position, heading, plan, boxes, backoff, arrival_tolerance, yaw_tolerance=TRIAL_YAW_TOLERANCE):
        """Return the current or a farther standing offset on the same face normal."""
        direction = np.asarray(plan.direction)
        error = (math.atan2(direction[1], direction[0]) - heading + math.pi) % (2*math.pi) - math.pi
        end = heading if abs(error) < yaw_tolerance else heading + error
        if body_rotation_clear(self, boxes, position, heading, end):
            return backoff
        for offset in (.25, .45):
            stage = np.asarray(plan.approach_xy) - offset*direction
            if (offset > backoff and np.linalg.norm(stage-position[:2]) > arrival_tolerance
                    and body_rotation_clear(self, boxes, stage, heading, end)):
                return offset
        return None

    def first_contact_plan(self, start, boxes, eligible, free, profile, *, upper_end_goal=None, guidance_boxes=None):
        """Select a reachable face, optionally reserving the map's upper short end."""
        best = None
        for tracked in boxes:
            if tracked.id not in eligible:
                continue
            if upper_end_goal is not None and (tracked.orientation is None or tracked.local_dimensions is None):
                continue
            # Use the stable pose to select a face and a standing point. Route
            # and body-sweep checks still receive the current obstacle boxes.
            guide = (guidance_boxes or {}).get(tracked.id)
            box = self.oriented_box(guide if guide is not None else tracked)
            center = np.asarray(box.xy)
            for i, direction in enumerate(self.push_directions(box)):
                axis = i // 2
                if upper_end_goal is not None:
                    # An end face is normal to the long horizontal dimension.
                    if box.dimensions[axis] <= box.dimensions[1-axis] or -direction[1] <= abs(direction[0]):
                        continue
                elif trial_push_speed(self, box, direction, boxes, free, profile) is None:
                    continue
                extent = box.dimensions[axis] / 2
                approach = center - direction*(extent + profile.nose_offset_m + PushAsideConfig.front_gap)
                path = self.path(start, approach, boxes)
                if upper_end_goal is not None:
                    approach, path = self._upper_turn_stage(start, boxes, free, profile, approach, direction, path,
                                                            min(tracked.local_dimensions[:2]))
                if not path:
                    continue
                score = self.length(start, path)
                if best is None or score < best.score:
                    distance = SHORT_END_PUSH_DISTANCE_M if upper_end_goal is not None else TRIAL_DISTANCE_M
                    best = PushPlan(box.id, box.xy, tuple(center + direction*distance),
                                    tuple(direction), tuple(approach), path, distance, score)
        return best

    def alternate_upper_stage(self, start, boxes, profile, plan, excluded):
        target = next((box for box in boxes if box.id == plan.box_id), None)
        if target is None or target.local_dimensions is None:
            return None, ()
        direction = np.asarray(plan.direction)
        distance = max(target.local_dimensions[:2])/2 + profile.nose_offset_m + PushAsideConfig.front_gap
        anchor = np.asarray(plan.initial_xy) - distance*direction
        return self._upper_turn_stage(start, boxes, self.free, profile, anchor, direction,
                                      self.path(start, anchor, boxes), min(target.local_dimensions[:2]), excluded=excluded)

    def _upper_turn_stage(self, start, boxes, free, profile, anchor, direction, center_path, face_width, *, excluded=()):
        # Use the same footprint as the turn controller. If the center is too
        # tight to turn, the existing skill can center laterally after turning.
        check = copy(self)
        check.free = free
        SdkNavigationMotion(profile).bind(check)
        tangent = np.array([-direction[1], direction[0]])
        target_yaw = math.atan2(direction[1], direction[0])
        # Walk to the centerline at a slightly greater standoff before asking
        # the field robot for a 32 cm sidestep. All three gaps allow engagement.
        candidates = ((0., 0.), (0., .10), (0., .20), (-.10, 0.), (.10, 0.),
                      (-.20, 0.), (.20, 0.), (-MAX_SIDE_STAGE_OFFSET_M, 0.), (MAX_SIDE_STAGE_OFFSET_M, 0.))
        if excluded:
            # After a real gait stall, try arriving already within the face
            # band at different stand-offs before asking for another sidestep.
            near_face = [(offset, backoff) for backoff in (0., .10, .20, -.10)
                         for offset in (0., -SHORT_END_CENTER_BAND_M, SHORT_END_CENTER_BAND_M)]
            candidates = dict.fromkeys([*near_face, *candidates, (-.15, -.10), (.15, -.10)])
        for offset, backoff in candidates:
            center = anchor - backoff * direction
            stage = center + offset*tangent
            if any(np.linalg.norm(stage - xy) <= SIDE_STAGE_ARRIVAL_M for xy in excluded):
                continue
            path = center_path if not (offset or backoff) else self.path(start, stage, boxes)
            if not path:
                continue
            previous = path[-2][:2] if len(path) > 1 else start[:2]
            delta = stage - previous
            yaw = math.atan2(delta[1], delta[0])
            arrival = SIDE_STAGE_ARRIVAL_M if offset else NEARBY_TURN_RADIUS_M
            if np.linalg.norm(stage - start[:2]) <= arrival and self.robot_heading is not None:
                yaw = self.robot_heading
            end = yaw + (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            if body_rotation_clear(check, boxes, stage, yaw, end) and (
                    not offset or self._centering_path_clear(check, boxes, stage, target_yaw, tangent,
                                                            -offset, face_width, profile)):
                return stage, path
        # A scan from nearer the box can reveal the turn. Recheck the full
        # rotation at the actual stopped pose before issuing any yaw command.
        return (None, ()) if excluded else (anchor, center_path)

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
    review_timeout_s = 8.
    stall_seconds = 2.5
    max_lateral_drift_m = .06

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.near_at = self.stop_at = None
        self.progress_mark_m = 0.
        self.last_progress_at = None
        self.resume_after = None
        self.start_base = self.stop_box = self.stop_base = None
        self.maximum_travel = None
        self.review = ObservedBoxMotion(self.backend.profile)
        self.measurement = {}
        self.precontact_wait = None
        self.guidance_box = None
        self.recheck_stopped_at = None
        self.recheck_wide_since = None
        self.timed_push = None

    def _lateral_alignment_error(self, ctx):
        if self.contact_started is None and self.guidance_box is not None:
            ctx = replace(ctx, box=self.guidance_box)
        forward = np.array([math.cos(ctx.heading), math.sin(ctx.heading)])
        gap, lateral, yaw_error = self._box_frame(ctx, travel_direction=forward)
        limit = self.config.heading_drift_tolerance if self.backend.upper_short_end else self.config.yaw_tolerance
        if abs(yaw_error) > limit or not math.isfinite(gap):
            return super()._lateral_alignment_error(ctx)
        # With an accepted residual yaw, center the nose and its projected
        # landing point together. Both must fit the selected end face.
        return lateral - .5 * max(0., gap) * float(forward @ self.tangent)

    def _contact_ready(self, ctx):
        if not self.backend.upper_short_end:
            return super()._contact_ready(ctx)
        if ctx.box_observation_id is None:
            return False
        forward = np.array([math.cos(ctx.heading), math.sin(ctx.heading)])
        gap, _, yaw = self._box_frame(ctx, travel_direction=forward)
        return (.20 <= gap <= self.config.max_engage_standoff and abs(yaw) <= self.config.yaw_tolerance
                and self._contact_face_clear(ctx))

    def _push_speed(self, elapsed, attempt_progress):
        speed = super()._push_speed(elapsed, attempt_progress)
        # Ramp back to engagement speed for the final centimetres. At peak
        # speed a single scan interval can advance the box past the target.
        remaining = max(0., self.stop_displacement - attempt_progress)
        profile = self.backend.profile
        approach_speed = math.sqrt(2 * profile.linear_acceleration_mps2 * remaining)
        return min(speed, max(profile.push_start_mps, approach_speed))

    def _selected_face(self, box):
        axes = rotation(box.orientation)[:2, :2]
        alignment = self.direction @ axes
        axis = int(np.argmax(abs(alignment)))
        return axes[:, axis]*np.sign(alignment[axis]), box.local_dimensions[axis]/2

    def _box_frame(self, ctx, travel_direction=None):
        # Intersect the nose's line of travel with the real oriented face. The
        # padded world AABB used for route clearance can be several cm closer.
        normal, half_depth = self._selected_face(ctx.box)
        nose = (ctx.position + rotation(ctx.orientation) @ [self.backend.nose_offset_m, 0., -.06])[:2]
        delta = np.asarray(ctx.box.xy)-nose
        toward = normal @ (self.direction if travel_direction is None else travel_direction)
        gap = (delta @ normal-half_depth)/toward if toward > 0. else math.inf
        yaw_error = (self.push_heading-ctx.heading+math.pi) % (2*math.pi)-math.pi
        return float(gap), float(delta @ self.tangent), yaw_error

    def landing_offset(self, ctx, forward, gap):
        normal, _ = self._selected_face(ctx.box)
        tangent = np.array([-normal[1], normal[0]])
        nose = (ctx.position + rotation(ctx.orientation) @ [self.backend.nose_offset_m, 0., -.06])[:2]
        return float((nose + max(0., gap)*forward - ctx.box.xy) @ tangent)

    def _fail(self, ctx, reason):
        checker = getattr(getattr(ctx, 'layers', None), 'surface_checker', None)
        if (self.backend.upper_short_end and self.state in ('engage', 'push') and reason == 'contact_motion_blocked'
                and checker is not None and checker.diagnostic['reason'] == 'box_edge_needs_new_scan'):
            self.precontact_wait = 'box_edge_needs_new_scan'
            return self._output()
        if (self.backend.upper_short_end and self.contact_started is None
                and self.state in ('align_lateral', 'settle_contact')
                and reason in ('lid_centering_no_progress', 'sdk_alignment_stalled', 'sdk_body_or_box_sweep_blocked')):
            self._transition('recheck_lid', ctx, reason)
            self.precontact_wait = 'braking_for_lid_recheck'
            self.backend.centering = None
            return self._output()
        return super()._fail(ctx, reason)

    def _recheck_lid(self, ctx):
        if not ctx.healthy or ctx.tilt_degrees >= self.backend.hard_tilt_limit:
            return self._fail(ctx, 'robot_unstable')
        if self.started is not None and ctx.now - self.started > self.config.max_skill_seconds:
            return self._fail(ctx, 'skill_timeout')
        if not self._quiet(ctx, .4):
            self.recheck_stopped_at = None
            self.precontact_wait = 'braking_for_lid_recheck'
        else:
            if self.recheck_stopped_at is None:
                self.recheck_stopped_at = ctx.now
                self.events.append(dict(event='lid_recheck_stopped', time_s=ctx.now,
                                        position_xy=ctx.position[:2].tolist(), heading_rad=ctx.heading))
            if (not ctx.box_top_observed or ctx.box_observation_id is None or ctx.box_observed_at is None
                    or not self.recheck_stopped_at < ctx.box_observed_at <= ctx.now):
                self.precontact_wait = 'waiting_for_stopped_lid'
            elif ctx.map_observed_at is None or not self.recheck_stopped_at < ctx.map_observed_at <= ctx.now:
                self.precontact_wait = 'waiting_for_push_map'
            else:
                self.precontact_wait = self._check_straight_push(ctx)
        if self.precontact_wait == 'stopped_head_wide_of_face':
            if self.recheck_wide_since is None:
                self.recheck_wide_since = ctx.box_observed_at
            if ctx.box_observed_at - self.recheck_wide_since >= self.backend.profile.box_settle_seconds:
                return self._fail(ctx, 'confirmed_landing_outside_face')
        else:
            self.recheck_wide_since = None
        if self.precontact_wait and ctx.now - self.state_since >= self.config.max_alignment_seconds:
            return self._fail(ctx, self.precontact_wait)
        return self._output()

    def _check_straight_push(self, ctx):
        forward = np.array([math.cos(ctx.heading), math.sin(ctx.heading)])
        gap, _, yaw = self._box_frame(ctx, travel_direction=forward)
        if abs(yaw) > self.config.yaw_tolerance or not .20 <= gap <= self.config.max_engage_standoff:
            return 'stopped_pose_outside_standoff_or_heading'
        if not self._contact_face_clear(ctx):
            if abs(self.landing_offset(ctx, forward, gap)) > min(ctx.box.local_dimensions[:2])/2:
                return 'stopped_head_wide_of_face'
            return 'stopped_head_misses_short_face'
        self.engage_box_position = np.array(ctx.box.xy)
        self._transition('engage', ctx, 'stopped_lid_allows_straight_push')
        self.events[-1].update(stopped_at=self.recheck_stopped_at, scan_at=ctx.box_observed_at,
                               nose_gap_m=gap, landing_offset_m=self.landing_offset(ctx, forward, gap))
        return None

    def stop_trial(self, now, position, reason, box=None):
        self.stop_at = now
        self.stop_base = np.array(position[:2])
        self.stop_box = None if box is None else np.array(box.xy)
        self.status, self.state, self.reason = 'checking', 'first_push_stop', reason
        return self._output()

    def box_within_stopping_reach(self, ctx, command):
        if not self.backend.upper_short_end or self.state not in ('recheck_lid', 'engage'):
            return True
        p = self.backend.profile
        speed = max(np.linalg.norm(command[:2]), np.linalg.norm(ctx.body_velocity[:2]),
                    np.linalg.norm(ctx.limiter.measured[:2]))
        # Bound the entire rotating body with its foot radius; measure the
        # distance to the actual oriented rectangle, including fit uncertainty.
        local = (ctx.position[:2] - ctx.box.xy) @ rotation(ctx.box.orientation)[:2, :2]
        nearest = np.maximum(0., abs(local) - np.asarray(ctx.box.local_dimensions[:2]) / 2)
        radius = math.hypot(max(-p.foot_x_min_m, p.foot_x_max_m), p.foot_half_width_m)
        clearance = np.linalg.norm(nearest) - radius - p.release_margin_m
        return clearance <= p.stopping_distance(speed, pushing=True, command_speed=speed)

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
        self.precontact_wait = None
        if self.state == 'recheck_lid':
            return self._recheck_lid(ctx)
        if self.state == 'settle_contact':
            if ctx.box_observation_id is not None:
                reason = self.backend.observe_box(self, ctx)
                if reason:
                    return self._fail(ctx, reason)
            if ctx.box_observation_id is None or ctx.box_observed_at <= self.state_since:
                self.precontact_wait = 'waiting_for_box_face'
            elif not self.backend.box_motion.settled:
                self.precontact_wait = 'waiting_for_settled_box'
            elif ctx.map_observed_at is None or ctx.map_observed_at <= self.state_since:
                self.precontact_wait = 'waiting_for_push_map'
            elif not self.backend.upper_short_end:
                speed = trial_push_speed(ctx.planner, ctx.box, self.direction, ctx.boxes,
                                         ctx.planner.free & ctx.planner.box_free, self.backend.profile, self.stop_displacement)
                if speed is None:
                    self.precontact_wait = 'push_path_blocked_or_unknown'
                elif speed != self.config.push_max_speed:
                    self.config = replace(self.config, push_max_speed=speed)
            if self.precontact_wait:
                if ctx.now - self.state_since >= self.config.max_alignment_seconds:
                    return self._fail(ctx, self.precontact_wait)
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
            if abs(delta @ self.tangent) >= self.max_lateral_drift_m or along < -.02:
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
                # A shifted mean must also exceed the measured scatter of the
                # final stationary window before a short push counts as motion.
                minimum = max(.02, self.review.diagnostic['max_surface_displacement_m'])
                self.measurement = dict(displacement_m=distance, lateral_m=lateral, min_displacement_m=minimum,
                    body_coast_m=float(np.linalg.norm(position[:2]-self.stop_base)),
                    box_coast_m=None if self.stop_box is None else float((center-self.stop_box) @ self.direction),
                    stop_reason=self.reason, final_scan_id=str(scan_id), final_scan_at=stamp)
                moved = distance >= minimum and abs(lateral) < self.max_lateral_drift_m and self.reason != 'box_moved_off_push_line'
                self.reason = 'first_push_observed' if moved else 'first_push_displacement_unconfirmed'
                self.status = 'succeeded' if moved else 'failed'
                self.last_box = np.array(box.xy)
                return moved
        if now-self.stop_at >= self.review_timeout_s:
            self.measurement = dict(stop_reason=self.reason)
            self.status, self.reason = 'failed', 'first_push_waiting_for_final_scan'
            return False
        return None

    def report(self):
        report = {**super().report(), 'first_push': dict(self.measurement),
                'selected_push_speed_mps': self.config.push_max_speed if self.contact_started is not None else None}
        if self.timed_push is not None:
            push = self.timed_push
            report['selected_push_speed_mps'] = push.speed
            report['timed_push'] = dict(nose_gap_m=push.gap, target_push_m=push.distance,
                                        speed_mps=push.speed, approach_speed_mps=push.approach_speed, duration_s=push.duration,
                                        planned_at=push.planned_at, sdk_started_at=push.sdk_started_at,
                                        approach_timeout_s=push.approach_timeout, face_reached_at=push.face_reached_at,
                                        stroke_seconds=push.stroke_seconds)
        return report
