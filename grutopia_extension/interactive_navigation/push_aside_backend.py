"""Execution seam for the single shared push state machine.

PushAsideSkill owns phases, route selection, retries and passage geometry. This
backend owns motion response and observation interpretation. The default below
preserves the released simulation behavior. Device adapters are injected by the
caller; this module never imports deployment code or selects a device by name.
"""
import numpy as np
from .go2_contact_control import ContactYawServo, contact_face_lateral_allowance, push_braking_distance
from .go2_contact_turn import ContactTurnControl
from .go2_heading_trim import HeadingTrimControl
from .go2_passage_confirmation import observed_passage_routes


class PushAsideBackend:
    policy_translation_steps = True
    nose_offset_m = .33
    movement_tilt_limit = 18.
    hard_tilt_limit = 30.

    def validate_permission(self, permission):
        if permission.source == 'rgb_dino_qwen_vl':
            if len(set(permission.evidence_frames)) < 2 or not permission.classifier_models:
                raise ValueError('Vision mode requires two independent recognized frames')
        elif permission.source == 'ground_truth_layout':
            if (not permission.source_reference or len(permission.source_digest) != 64
                    or any(c not in '0123456789abcdef' for c in permission.source_digest)
                    or permission.evidence_frames or permission.classifier_models):
                raise ValueError('Ground-truth mode requires explicit provenance without fake visual evidence')
        else:
            raise ValueError('Unsupported interaction permission source')

    def yaw_servo(self, tolerance):
        return ContactYawServo(tolerance)

    def heading_trim(self, skill):
        return HeadingTrimControl(skill.push_heading, min(skill.config.alignment_yaw_tolerance, skill.config.yaw_tolerance),
                                  acceptance_tolerance=skill.config.yaw_tolerance)

    def new_turn(self, skill, ctx):
        return ContactTurnControl(ctx.position, ctx.heading, skill.push_heading, ctx.planner, ctx.boxes, ctx.now)

    def update_turn(self, turn, ctx):
        return turn.update(ctx.position, ctx.heading, ctx.planner, ctx.boxes, ctx.limiter, ctx.now)

    def quiet_requirements(self, seconds, max_tilt):
        return seconds, max_tilt

    def contact_face_clear(self, skill, ctx, *, moving=False):
        return skill._default_contact_face_clear(ctx, moving=moving)

    def lateral_allowance(self, skill, ctx, *, moving=False):
        maximum = skill.config.pushing_face_offset_max if moving else skill.config.preparation_face_offset_max
        error = (skill.push_heading-ctx.heading+np.pi) % (2*np.pi)-np.pi
        return contact_face_lateral_allowance(ctx.box.dimensions, skill.direction, error, ctx.tilt_degrees,
                                            maximum, skill.config.contact_face_margin)

    def translate(self, skill, ctx, target, axis, speed, tolerance):
        return skill._policy_translation(ctx, target, axis, speed, tolerance)

    def before_step(self, skill, ctx):
        return None

    def after_step(self, skill, ctx, command):
        return command

    def observe_box(self, skill, ctx):
        if skill.last_box is not None and ctx.now > skill.last_box_time:
            if np.linalg.norm(np.asarray(ctx.box.xy)-skill.last_box) > 1e-5:
                skill.last_box_motion_time = ctx.now
            observed = float(np.dot(np.asarray(ctx.box.xy)-skill.last_box, skill.direction))/(ctx.now-skill.last_box_time)
            skill.box_velocity = .8*skill.box_velocity+.2*observed
        return None

    def passage_evidence(self, skill, ctx):
        return {'confirming_observation_time_s': float(getattr(ctx.layers, 'last_local', -1.))}

    def verify_passage(self, skill, ctx, tracked, along):
        proof, route, full = observed_passage_routes(ctx.layers, tracked, skill.task_start, skill.task_goal,
                                                   np.asarray(ctx.position[:2]), ctx.layers.current_boxes)
        return proof if proof and route and full else None

    def requires_braking_reserve(self, ctx):
        return getattr(ctx, 'policy_command', None) is not None

    def recovery_reverse_speed(self, ctx):
        return .45 if getattr(ctx, 'policy_command', None) is not None else .35

    def recovery_horizon(self, measured):
        return .8

    def retreat_options(self, skill, ctx):
        return {'speed': skill.config.reverse_speed} if getattr(ctx, 'policy_command', None) is not None else {}

    def push_stopping_distance(self, skill, ctx, velocity, rotation):
        return push_braking_distance(velocity)

    def contact_timeout(self, phase):
        return 12. if phase == 'engage' else 6.

    def stalled(self, skill, ctx, reason):
        return skill._retry(ctx, reason)

    def limit_request(self, request):
        return request

    def brake_failure(self, skill, ctx):
        return None

    def brake_complete(self, skill, ctx):
        return skill._quiet(ctx, .4, max_tilt=self.movement_tilt_limit) or ctx.now-skill.state_since > 1.5

    def feedback(self):
        return {}

    def report(self, skill):
        return {'object_pose_feedback': 'selected permitted simulator rigid body'}
