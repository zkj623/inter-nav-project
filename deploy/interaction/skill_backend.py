"""SDK execution adapter for the shared PushAsideSkill state machine.

Only this module interprets measured SDK response and registered LiDAR evidence.
It supplies primitives and guards; it does not own or copy the skill phase loop.
"""
from dataclasses import replace
import math
import numpy as np
from grutopia_extension.interactive_navigation.push_aside_backend import PushAsideBackend
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill, PushAsideConfig
from grutopia_extension.interactive_navigation.go2_contact_control import contact_heading_hold, contact_face_lateral_allowance, observed_contact_face_clear
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear, body_translation_recovery_clear
from .box_motion import ObservedBoxMotion
from .motion import SdkYawServo, SdkHeadingTrim


def wrap(value):
    return (value+math.pi)%(2*math.pi)-math.pi


def create_push_skill(permission, plan, start, goal, profile, *, config=None):
    settings = replace(config or PushAsideConfig(), **profile.skill_overrides())
    return PushAsideSkill(permission, plan, start, goal, settings, backend=SdkPushBackend(profile))


class SdkPushBackend(PushAsideBackend):
    policy_translation_steps = False

    def __init__(self, profile):
        self.profile = profile
        self.nose_offset_m = profile.nose_offset_m
        self.movement_tilt_limit = self.hard_tilt_limit = profile.max_tilt_degrees
        self.box_motion = ObservedBoxMotion(profile)

    def validate_permission(self, permission):
        if permission.source == 'ground_truth_layout':
            raise ValueError('SDK contact cannot use simulated ground-truth permission')
        frames = permission.evidence_frames
        if (permission.source != 'registered_box_lidar'
                or not isinstance(permission.source_reference, str) or not permission.source_reference.strip()
                or not isinstance(permission.source_digest, str) or len(permission.source_digest) != 64
                or any(c not in '0123456789abcdef' for c in permission.source_digest)
                or len(set(frames)) < 2 or any(not isinstance(frame, str) or not frame.startswith('lidar:') for frame in frames)
                or permission.classifier_models):
            raise ValueError('Registered LiDAR permission requires fresh LiDAR provenance without fake visual evidence')

    def yaw_servo(self, tolerance):
        return SdkYawServo(tolerance, self.profile)

    def heading_trim(self, skill):
        return SdkHeadingTrim(skill.push_heading, min(skill.config.alignment_yaw_tolerance, skill.config.yaw_tolerance),
                              acceptance_tolerance=skill.config.yaw_tolerance, profile=self.profile)

    def new_turn(self, skill, ctx):
        return SdkHeadingTrim(skill.push_heading, skill.config.yaw_tolerance, profile=self.profile)

    def update_turn(self, turn, ctx):
        angular = np.asarray(ctx.body_angular_velocity if ctx.body_angular_velocity is not None else [0., 0., ctx.yaw_rate])
        command, done, detail = turn.update(ctx.position, ctx.heading, ctx.planner, ctx.boxes,
                                           ctx.limiter, ctx.now, ctx.body_velocity, angular, ctx.tilt_degrees)
        return None if done else (command, detail)

    def quiet_requirements(self, seconds, max_tilt):
        return max(seconds, self.profile.settle_seconds), min(max_tilt, self.profile.max_tilt_degrees)

    def lateral_allowance(self, skill, ctx, *, moving=False):
        maximum = skill.config.pushing_face_offset_max if moving else skill.config.preparation_face_offset_max
        return contact_face_lateral_allowance(ctx.box.dimensions, skill.direction,
            wrap(skill.push_heading-ctx.heading), ctx.tilt_degrees, maximum, skill.config.contact_face_margin,
            head_bounds=self.profile.head_bounds, nose_offset=self.profile.nose_offset_m)

    def contact_face_clear(self, skill, ctx, *, moving=False):
        geometry = [getattr(ctx.box, key, None) for key in ('local_dimensions', 'orientation', 'center_z')]
        if any(value is None for value in geometry) or getattr(ctx, 'orientation', None) is None:
            return False
        maximum = skill.config.pushing_face_offset_max if moving else skill.config.preparation_face_offset_max
        return observed_contact_face_clear(ctx.position, ctx.orientation, ctx.box.xy, geometry[2],
            geometry[0], geometry[1], skill.direction, maximum, skill.config.contact_face_margin,
            head_bounds=self.profile.head_bounds, require_vertical=True)

    def before_step(self, skill, ctx):
        if ctx.box is None or ctx.box.id != skill.plan.box_id:
            return 'target_track_lost'
        if getattr(getattr(ctx.planner, 'motion_backend', None), 'profile', None) != self.profile:
            return 'contact_profile_mismatch'
        if (not np.isfinite([ctx.now, ctx.heading, ctx.yaw_rate, ctx.tilt_degrees]).all()
                or not np.isfinite(ctx.position).all() or not np.isfinite(ctx.body_velocity).all()):
            return 'invalid_measured_motion'
        if skill.contact_started is not None and ctx.now-skill.contact_started >= self.profile.max_contact_seconds:
            return 'contact_budget_exhausted'
        if (skill.state in {'align_longitudinal', 'align_lateral'} and skill.progress_since is not None
                and ctx.now-skill.progress_since >= self.profile.stall_seconds
                and np.linalg.norm(np.asarray(ctx.position[:2])-skill.progress_mark) < .01):
            return 'sdk_alignment_stalled'
        return None

    def after_step(self, skill, ctx, result):
        if result.status != 'running':
            return result
        bounded = self.profile.limit(result.velocity)
        if np.any(np.abs(bounded) > 1e-9):
            contact = skill.state in {'engage', 'push', 'settle_push_progress', 'brake_after_push'}
            if skill.contact_started is not None:
                check = getattr(ctx, 'contact_motion_check', None)
                if not callable(check):
                    return skill._fail(ctx, 'contact_motion_check_missing')
                try:
                    accepted = bool(check(bounded, ctx))
                except Exception:
                    accepted = False
                if not accepted:
                    return skill._fail(ctx, 'contact_motion_blocked')
            if not self.motion_clear(skill, ctx, bounded, contact=contact):
                return skill._fail(ctx, 'sdk_body_or_box_sweep_blocked')
        result.velocity = tuple(map(float, bounded))
        return result

    def observe_box(self, skill, ctx):
        try:
            self.box_motion.observe(ctx.box, ctx.box_observed_at, ctx.box_observation_id, ctx.now)
        except (TypeError, ValueError):
            return 'box_motion_observation_invalid'
        skill.last_box_motion_time = self.box_motion.last_motion_at
        skill.box_velocity = float(np.dot(self.box_motion.center_velocity, skill.direction))
        return None

    def passage_evidence(self, skill, ctx):
        stamp, observed = ctx.map_observed_at, self.box_motion
        if (not observed.settled or stamp is None or ctx.map_observation_id is None
                or not np.isfinite(stamp) or stamp > ctx.now+1e-6
                or stamp < observed.latest_at or stamp <= observed.last_motion_at):
            return None
        return {'confirming_observation_time_s': float(stamp),
                'confirming_observation_id': str(ctx.map_observation_id),
                'box_motion_evidence': dict(observed.diagnostic)}

    def verify_passage(self, skill, ctx, tracked, along):
        return skill._observed_passage_effect(ctx, along, commit=True)

    def requires_braking_reserve(self, ctx):
        return True

    def recovery_reverse_speed(self, ctx):
        return self.profile.reverse_mps

    def recovery_horizon(self, measured):
        return self.profile.motion_horizon(measured)

    def retreat_options(self, skill, ctx):
        return {'speed': skill.config.reverse_speed}

    def push_stopping_distance(self, skill, ctx, velocity, rotation):
        requested = max(0., float(np.dot(rotation@np.asarray(ctx.limiter.command[:2]), skill.direction)))
        return self.profile.stopping_distance(velocity, pushing=True, command_speed=requested)

    def contact_timeout(self, phase):
        return self.profile.stall_seconds

    def stalled(self, skill, ctx, reason):
        return skill._fail(ctx, reason)

    def limit_request(self, request):
        return self.profile.limit(request)

    def brake_failure(self, skill, ctx):
        return 'contact_braking_not_settled' if ctx.now-skill.state_since > skill.config.max_push_braking_seconds else None

    def brake_complete(self, skill, ctx):
        return skill._quiet(ctx, .4, max_tilt=self.movement_tilt_limit)

    def feedback(self):
        return {'box_motion_evidence': dict(self.box_motion.diagnostic)}

    def report(self, skill):
        return {'contact_backend': 'sdk', 'contact_profile': self.profile.to_dict(),
                'contact_started_at': skill.contact_started, **self.feedback(),
                'object_pose_feedback': 'live registered LiDAR geometry'}

    def translate(self, skill, ctx, target, axis, speed, tolerance):
        delta = np.asarray(target) - np.asarray(ctx.position[:2])
        error = float(np.dot(delta, axis))
        if skill.translation_braking:
            if not skill._quiet(ctx):
                return (np.zeros(3), False)
            skill.translation_braking = False
            return (np.zeros(3), abs(error) < tolerance)
        if abs(error) < tolerance:
            skill.translation_braking = True
            return (np.zeros(3), skill._quiet(ctx))
        skill.settled_since = None
        (c, s) = (math.cos(ctx.heading), math.sin(ctx.heading))
        rot = np.array([[c, -s], [s, c]])
        world_v = rot @ np.asarray(ctx.body_velocity[:2])
        unit = axis * math.copysign(1.0, error)
        toward = float(np.dot(world_v, unit))
        commanded = np.asarray(getattr(ctx.limiter, 'command', (0.0, 0.0, 0.0)), float)
        command_toward = max(0.0, float(np.dot(rot @ commanded[:2], unit)))
        stopping_distance = self.profile.stopping_distance(toward, command_speed=command_toward)
        if toward > 0.06 and abs(error) < stopping_distance:
            skill.translation_braking = True
            return (np.zeros(3), False)
        speed = float(np.clip(1.5 * abs(error) - 0.6 * max(0.0, toward), self.profile.min_translation_mps, speed))
        world = unit * speed
        yaw_request = contact_heading_hold(wrap(skill.push_heading - ctx.heading), ctx.yaw_rate)
        if not body_motion_clear(ctx.planner, ctx.boxes, ctx.position, ctx.heading, 0.0, 0.0, horizon=0.0):
            yaw_request = 0.0
        request = [*rot.T @ world, yaw_request]
        request = self.profile.limit(request)
        (command, _) = checked_contact_turn_command(request, ctx.position, ctx.heading, ctx.planner, ctx.boxes, ctx.limiter, ctx.now, target_position=np.asarray(ctx.position[:2]) + axis * error)
        return (command, False)

    def motion_clear(self, skill,ctx,command,*,contact):
        """Keep map obstacles and unknown space in every executed SDK sweep.

        The deployment callback independently checks unmasked current returns
        and target-surface ownership; omitting the target object layer here
        cannot authorize deletion of unrelated map cells.
        """
        boxes=[box for box in ctx.boxes if not (contact and box.id==ctx.box.id)]
        measured=np.asarray(ctx.limiter.measured,float)
        if measured.shape!=(3,) or not np.isfinite(measured).all():return False
        for velocity in (command,measured):
            horizon=self.profile.motion_horizon(velocity)
            safe=body_motion_clear(ctx.planner,boxes,ctx.position,ctx.heading,
                velocity[0],velocity[2],horizon=horizon,lateral=velocity[1])
            if not safe and not contact and abs(velocity[2])<.10 and abs(command[2])<1e-8:
                c,s=math.cos(ctx.heading),math.sin(ctx.heading)
                target=np.asarray(ctx.position[:2])+np.array([[c,-s],[s,c]])@velocity[:2]*horizon
                safe=body_translation_recovery_clear(ctx.planner,boxes,ctx.position,ctx.heading,target,require_clear_end=False)
            if not safe:return False
        if contact:
            # Reserve the box's complete swept extent, including measured
            # stopping travel. Recompute on every scan/command, not just plan.
            c,s=math.cos(ctx.heading),math.sin(ctx.heading)
            rotation=np.array([[c,-s],[s,c]])
            speed=max(0.,float(np.dot(rotation@command[:2],skill.direction)),
                float(np.dot(rotation@measured[:2],skill.direction)),skill.box_velocity)
            distance=self.profile.stopping_distance(speed,pushing=True,command_speed=speed)
            allowed=ctx.planner.box_free.copy()
            for box in boxes:
                mask=ctx.planner.box_mask(box,padding=self.profile.stop_margin_m)
                if mask is None:return False
                allowed &= ~mask
            for offset in np.linspace(0.,distance,max(2,int(math.ceil(distance/ctx.planner.resolution))+1)):
                mask=ctx.planner.box_mask(ctx.box,xy=np.asarray(ctx.box.xy)+offset*skill.direction,
                    padding=self.profile.stop_margin_m)
                if mask is None or not allowed[mask].all():return False
        return True
