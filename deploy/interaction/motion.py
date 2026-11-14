"""Measured SDK motion primitives used by shared route and skill algorithms."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.body_geometry import BodyFootprint
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear, body_translation_recovery_clear
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command


class SdkNavigationMotion:
    def __init__(self, profile):
        self.profile = profile

    def bind(self, planner):
        p = self.profile
        planner.motion_backend = self
        planner.body_geometry = BodyFootprint(p.foot_x_min_m, p.foot_x_max_m, p.foot_half_width_m)
        planner.release_braking_margin = p.release_margin_m
        planner.contact_offset = p.foot_x_max_m+.05

    def corner_maneuver(self, position, heading, planner, boxes):
        return None

    def local_action(self, action, state, position, heading, planner, boxes):
        bounded = self.profile.limit(action['move_by_speed'])
        safe = body_motion_clear(planner, boxes, position, heading, bounded[0], bounded[2],
                                 horizon=self.profile.motion_horizon(bounded), lateral=bounded[1])
        if not safe:
            return {'move_by_speed': [0., 0., 0.]}, {**state, 'needs_replan': True, 'sdk_motion_blocked': True}
        return {'move_by_speed': bounded.tolist()}, {**state, 'contact_backend': 'sdk'}

    def retreat_horizon(self, speed, acceleration):
        return self.profile.translation_stop_seconds+speed/(2*acceleration)

    def checked_contact_command(self,request, position, heading, planner, boxes, limiter, now, target_position=None):
        """Check both acceleration-limited requests and measured stopping motion.

        All boxes remain obstacles here: deliberate contact starts only after
        preparation. A stationary yaw request is not a stationary physical turn.
        """
        profile=self.profile
        request=profile.limit(request)
        command = limiter.preview(request, now)
        command=profile.limit(command)
        measured = limiter.measured
        horizon = max(profile.motion_horizon(command), profile.motion_horizon(measured))
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        if target_position is not None:
            delta=np.asarray(target_position[:2])-np.asarray(position[:2])
            world=rotation@command[:2];speed=np.linalg.norm(world);distance=np.linalg.norm(delta)
            if speed>1e-8 and distance>1e-8:
                alignment=float(np.clip(np.dot(world,delta)/(speed*distance),-1.,1.))
                if abs(command[2])<1e-8 and alignment>.95:
                    horizon=min(horizon,(distance+.025)/speed)
                elif alignment>.995:
                    # Heading hold adds a small yaw to a finite sidestep. Check
                    # its complete arc to the goal, not indefinite motion past it.
                    # The conservative projected speed covers up to 0.1 rad of
                    # turning; larger turns retain the original full horizon.
                    cosine=math.cos(math.acos(alignment)+.10)
                    finite=(distance+.025)/(speed*cosine)
                    if abs(command[2])*finite<=.10:horizon=min(horizon,finite)
                # Actual stopping motion remains independently checked below.
        safe = body_motion_clear(planner, boxes, position, heading,
            command[0], command[2], horizon=horizon, lateral=command[1])
        # Staging may start in a mild existing overlap; allow only withdrawal.
        if not safe and abs(command[2]) < 1e-8:
            target = np.asarray(position[:2]) + rotation @ command[:2] * horizon
            safe = body_translation_recovery_clear(planner, boxes, position, heading,
                target, require_clear_end=False)
        moving = np.linalg.norm(measured[:2]) > .06 or abs(measured[2]) > .10
        if moving:
            measured_horizon=profile.motion_horizon(measured)
            momentum_safe = body_motion_clear(planner, boxes, position, heading,
                measured[0], measured[2], horizon=measured_horizon, lateral=measured[1])
            if not momentum_safe and abs(command[2]) < 1e-8 and abs(measured[2]) < .1:
                momentum_safe = body_translation_recovery_clear(planner, boxes, position, heading,
                    np.asarray(position[:2]) + rotation @ measured[:2] * measured_horizon, require_clear_end=False)
            safe = safe and momentum_safe
        return command if safe else np.zeros(3), bool(safe)

    def buffer_translation(self,planner,boxes,position,heading,target,*,reverse_speed=.35):
        """A reachable free pose must also admit an effective, stoppable stride."""
        profile=self.profile
        delta=np.asarray(target[:2],float)-np.asarray(position[:2],float)
        distance=float(np.linalg.norm(delta))
        if distance<.02 or not body_translation_recovery_clear(planner,boxes,position,heading,target):return None
        c,s=math.cos(heading),math.sin(heading)
        body=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])/distance
        axis_speed=(profile.lateral_mps if abs(body[1])>abs(body[0]) else
                    profile.reverse_mps if body[0]<0 else profile.forward_mps)
        speed=float(np.clip(1.5*distance,profile.min_translation_mps,axis_speed))
        command=profile.limit([*(speed*body),0.])
        travel=max(distance,profile.stopping_distance(np.linalg.norm(command[:2]),command_speed=np.linalg.norm(command[:2])))
        endpoint=np.asarray(position[:2])+delta/distance*travel
        if not body_translation_recovery_clear(planner,boxes,position,heading,endpoint,require_clear_end=False):return None
        return command


class SdkYawServo:
    """Finish a small heading correction with measured braking before reversal."""

    def __init__(self, tolerance=0.1, profile=None):
        self.profile = profile
        self.tolerance = float(tolerance)
        self.braking = False
        self.stable_since = None
        self.direction = 0.0

    def update(self, error, measured_rate, now):
        if self.braking:
            if abs(measured_rate) < 0.06:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= 0.2:
                    self.braking = False
                    self.direction = 0.0
                    self.stable_since = None
            else:
                self.stable_since = None
            return 0.0
        if self.direction == 0.0:
            self.direction = float(np.sign(error))
        toward = self.direction * measured_rate
        stopping_angle = max(0.0, toward) * self.profile.yaw_stop_seconds + self.profile.yaw_stop_margin_rad
        if abs(error) < 0.7 * self.tolerance or error * self.direction < 0 or (toward > 0.06 and abs(error) < stopping_angle):
            self.braking = True
            self.stable_since = None
            return 0.0
        rate = 3.0 * abs(error) - 0.35 * max(0.0, toward)
        return self.direction * float(np.clip(rate, self.profile.min_yaw_rps, self.profile.max_yaw_rps))


class SdkHeadingTrim:
    """Measured heading correction; no learned-policy pulses or gait priming."""
    def __init__(self, goal_heading, tolerance=.05, *, profile, max_translation=.25, acceptance_tolerance=None):
        self.profile, self.goal_heading = profile, float(goal_heading)
        self.tolerance = float(tolerance)
        self.acceptance_tolerance = float(tolerance if acceptance_tolerance is None else acceptance_tolerance)
        if self.acceptance_tolerance < self.tolerance:
            raise ValueError('Settled acceptance must contain the aiming band')
        self.max_translation = float(max_translation)
        self.servo = SdkYawServo(tolerance, profile)
        self.stop = MeasuredStopGate(hold_seconds=profile.settle_seconds)
        self.origin = self.best_error = self.last_progress = self.stop_heading = None
        self.braking, self.done, self.travel_budget_exhausted = True, False, False
        self.failed, self.last_detail = None, {}

    def update(self, position, heading, planner, boxes, limiter, now, body_velocity, angular_velocity, tilt, *, policy_command=None):
        position = np.asarray(position, float)
        error = (self.goal_heading-heading+math.pi)%(2*math.pi)-math.pi
        if self.origin is None:
            self.origin, self.best_error, self.last_progress = position[:2].copy(), abs(error), now
        travel = float(np.linalg.norm(position[:2]-self.origin))
        if travel > self.max_translation and not self.travel_budget_exhausted:
            self.travel_budget_exhausted = self.braking = True
            self.stop.reset()
            self.stop_heading = None
        if abs(error) < self.best_error-.015:
            self.best_error, self.last_progress = abs(error), now
        if not self.braking and now-self.last_progress >= self.profile.stall_seconds:
            self.failed = 'sdk_heading_stalled'
        detail = {'heading_error_rad': float(error), 'best_error_rad': float(self.best_error),
                  'translation_from_origin_m': travel, 'mode': 'yaw_trim',
                  'aiming_tolerance_rad': self.tolerance, 'settled_tolerance_rad': self.acceptance_tolerance,
                  'travel_budget_exhausted': self.travel_budget_exhausted, 'contact_backend': 'sdk'}

        def checked(request):
            command, safe = checked_contact_turn_command(self.profile.limit(request), position, heading, planner, boxes, limiter, now)
            detail.update(body_motion_safe=bool(safe), applied_command=command.tolist())
            self.last_detail = detail.copy()
            return command

        if self.failed or self.done:
            return checked([0., 0., 0.]), self.done, detail
        if abs(error) < self.tolerance and not self.braking:
            self.braking = True
            self.stop.reset()
            self.stop_heading = None
        if self.braking:
            detail['mode'] = 'heading_trim_brake'
            command = checked([0., 0., 0.])
            if np.linalg.norm(command[:2]) >= .03 or abs(command[2]) >= .06:
                self.stop.reset()
                self.stop_heading = None
            else:
                if self.stop_heading is None:
                    self.stop_heading = heading
                drift = abs((heading-self.stop_heading+math.pi)%(2*math.pi)-math.pi)
                detail['settling_yaw_drift_rad'] = float(drift)
                if drift > .015:
                    self.stop.reset()
                    self.stop_heading = heading
                if self.stop.update(position, body_velocity, angular_velocity, tilt, now):
                    outward = max(0., -float(np.sign(error))*float(angular_velocity[2]))
                    detail['predicted_outward_error_rad'] = float(abs(error)+outward)
                    if abs(error)+outward < self.acceptance_tolerance:
                        self.done = True
                    elif self.travel_budget_exhausted:
                        self.failed = 'heading_trim_translation_limit'
                    else:
                        self.braking = False
                        self.stop.reset()
                        self.stop_heading, self.last_progress = None, now
                        self.servo = SdkYawServo(self.tolerance, self.profile)
            self.last_detail = detail.copy()
            return command, self.done, detail
        rate = self.servo.update(error, float(angular_velocity[2]), now)
        if abs(rate) < 1e-9:
            self.braking = True
            self.stop.reset()
            self.stop_heading = None
            detail['mode'] = 'heading_trim_brake'
            return checked([0., 0., 0.]), False, detail
        return checked([0., 0., rate]), False, detail
