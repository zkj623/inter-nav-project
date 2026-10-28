"""Measured-stop heading trim with a checked stepping fallback for a stuck yaw gait."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import ContactYawServo
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear


class HeadingTrimControl:
    def __init__(self,goal_heading,tolerance=.05,stalled_seconds=2.,max_translation=.25,acceptance_tolerance=None):
        self.goal_heading=float(goal_heading);self.tolerance=float(tolerance)
        self.acceptance_tolerance=float(tolerance if acceptance_tolerance is None else acceptance_tolerance)
        if self.acceptance_tolerance<self.tolerance:raise ValueError('Settled acceptance must contain the aiming band')
        self.stalled_seconds=float(stalled_seconds);self.max_translation=float(max_translation)
        self.servo=ContactYawServo(tolerance);self.stop=MeasuredStopGate(hold_seconds=1.20)
        self.origin=None;self.best_error=None;self.last_progress=None
        self.braking=True;self.arc_enabled=False;self.arc_speed=None;self.done=False;self.failed=None
        self.last_detail={};self.stop_heading=None;self.travel_budget_exhausted=False
        self.arc_phase=None;self.prime_speed=None;self.prime_since=None;self.prime_stable_since=None;self.prime_origin=None;self.prime_direction=None

    def update(self,position,heading,planner,boxes,limiter,now,body_velocity,angular_velocity,tilt):
        position=np.asarray(position,float);error=(self.goal_heading-heading+math.pi)%(2*math.pi)-math.pi
        if self.origin is None:
            self.origin=position[:2].copy();self.best_error=abs(error);self.last_progress=now
        travel=float(np.linalg.norm(position[:2]-self.origin))
        if travel>self.max_translation and not self.travel_budget_exhausted:
            # Stop requesting travel at the budget, then assess the actual
            # settled heading. Braking drift is not a second motion request.
            self.travel_budget_exhausted=True;self.braking=True
            self.stop.reset();self.stop_heading=None;self.arc_phase=None
        if abs(error)<self.best_error-.015:self.best_error=abs(error);self.last_progress=now
        if now-self.last_progress>=self.stalled_seconds:self.arc_enabled=True
        detail={'heading_error_rad':float(error),'best_error_rad':float(self.best_error),
            'translation_from_origin_m':travel,'arc_enabled':self.arc_enabled,'mode':'yaw_trim',
            'aiming_tolerance_rad':self.tolerance,'settled_tolerance_rad':self.acceptance_tolerance,
            'travel_budget_exhausted':self.travel_budget_exhausted}
        def checked(request):
            command,safe=checked_contact_turn_command(request,position,heading,planner,boxes,limiter,now)
            detail.update(body_motion_safe=bool(safe),applied_command=command.tolist())
            self.last_detail=detail.copy()
            return command
        if self.failed or self.done:
            return checked([0.,0.,0.]),self.done,detail
        if abs(error)<self.tolerance and not self.braking:
            self.braking=True;self.stop.reset();self.stop_heading=None;self.arc_phase=None
        if self.braking:
            detail['mode']='heading_trim_brake';command=checked([0.,0.,0.])
            command_quiet=np.linalg.norm(command[:2])<.03 and abs(command[2])<.06
            if not command_quiet:
                self.stop.reset();self.stop_heading=None
            else:
                if self.stop_heading is None:self.stop_heading=heading
                drift=abs((heading-self.stop_heading+math.pi)%(2*math.pi)-math.pi)
                detail['settling_yaw_drift_rad']=float(drift)
                if drift>.015:
                    self.stop.reset();self.stop_heading=heading
                if self.stop.update(position,body_velocity,angular_velocity,tilt,now):
                    outward=max(0.,-float(np.sign(error))*float(angular_velocity[2]))
                    detail['predicted_outward_error_rad']=float(abs(error)+outward)
                    if abs(error)+outward<self.acceptance_tolerance:self.done=True
                    elif self.travel_budget_exhausted:self.failed='heading_trim_translation_limit'
                    else:
                        self.braking=False;self.stop.reset();self.stop_heading=None
                        self.servo=ContactYawServo(self.tolerance)
            self.last_detail=detail.copy()
            return command,self.done,detail
        if self.arc_enabled and self.arc_phase is None:
            probe_rate=math.copysign(1.,error)
            speeds=[self.arc_speed] if self.arc_speed is not None else [-.20,.20]
            for speed in speeds:
                prime_speed=-.35 if speed<0 else .30
                if (body_motion_clear(planner,boxes,position,heading,prime_speed,0.,horizon=.8)
                        and body_motion_clear(planner,boxes,position,heading,speed,probe_rate,horizon=.8)):
                    _,safe=checked_contact_turn_command([prime_speed,0.,0.],position,heading,planner,boxes,limiter,now)
                    if safe:
                        self.arc_speed=speed;self.prime_speed=prime_speed;self.arc_phase='prime';self.prime_since=now;self.prime_stable_since=None
                        self.prime_origin=position[:2].copy()
                        self.prime_direction=np.sign(speed)*np.array([math.cos(heading),math.sin(heading)])
                        break
            else:detail['arc_unavailable']=True
        if self.arc_phase=='prime':
            progress=float(np.dot(position[:2]-self.prime_origin,self.prime_direction))
            actual=float(np.sign(self.arc_speed)*body_velocity[0])
            commanded=float(np.sign(self.arc_speed)*limiter.command[0])
            stepping=commanded>=.9*abs(self.prime_speed) and actual>.06 and progress>.025
            if stepping:
                if self.prime_stable_since is None:self.prime_stable_since=now
            else:self.prime_stable_since=None
            detail.update(mode='prime_heading_step',prime_progress_m=progress,prime_speed_mps=actual)
            if self.prime_stable_since is not None and now-self.prime_stable_since>=.20:
                self.arc_phase='turn';self.servo=ContactYawServo(self.tolerance)
            elif now-self.prime_since>3.:
                self.failed='heading_trim_gait_not_started'
                return checked([0.,0.,0.]),False,detail
            else:
                return checked([self.prime_speed,0.,0.]),False,detail
        if self.arc_phase=='turn':
            measured=float(angular_velocity[2])
            stopping_angle=abs(measured)*.25+limiter.command[2]**2/(2*limiter.yaw_acceleration)+.03
            brake=(abs(error)<self.tolerance or
                (error*measured>0 and abs(measured)>.1 and abs(error)<stopping_angle) or
                (error*measured<0 and abs(measured)>.1))
            rate=0. if brake else math.copysign(1.,error)
        else:
            rate=self.servo.update(error,float(angular_velocity[2]),now)
        if abs(rate)<1e-9:
            self.braking=True;self.stop.reset();self.stop_heading=None;self.arc_phase=None;detail['mode']='heading_trim_brake'
            return checked([0.,0.,0.]),False,detail
        request=np.array([0.,0.,rate])
        if self.arc_phase=='turn':
            speeds=[self.arc_speed] if self.arc_speed is not None else [-.20,.20]
            for speed in speeds:
                if body_motion_clear(planner,boxes,position,heading,speed,rate,horizon=.8):
                    candidate,safe=checked_contact_turn_command([speed,0.,rate],position,heading,planner,boxes,limiter,now)
                    if safe:
                        self.arc_speed=speed;request=np.array([speed,0.,rate]);detail['mode']='checked_heading_step';detail['arc_speed_mps']=speed
                        break
            else:
                # End a blocked arc through a complete measured stop before
                # considering the opposite direction at the next correction.
                if self.arc_speed is not None:
                    self.arc_speed=None;self.braking=True;self.stop.reset();self.stop_heading=None;self.arc_phase=None
                    detail['mode']='heading_arc_blocked_brake';request=np.zeros(3)
                else:detail['arc_unavailable']=True
        return checked(request),False,detail
