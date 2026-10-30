"""Measured settling and continuous translation into a known safe turn pose."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver,turn_placement_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear,body_translation_recovery_clear
from grutopia_extension.interactive_navigation.go2_turn_response import TurnResponseAssist,stepping_arc_clear


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


class StagingYawRate:
    """Use the start request only until measured turning is established.

    A persistent 1 rad/s request can produce drift during a long in-place
    turn, even though yaw itself tracks. Keep the calibrated start request
    for a stalled stance, then cruise at .6 after sustained measured motion.
    The enclosing limiter and body sweep still check the applied command.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sign=None;self.moving_since=None;self.stalled_since=None
        self.cruising=False

    def request(self,error,measured_yaw,now):
        sign=math.copysign(1.,error)
        if sign!=self.sign:
            self.reset();self.sign=sign
        toward=sign*float(measured_yaw)
        if self.cruising:
            if toward<.15:
                if self.stalled_since is None:self.stalled_since=now
                if now-self.stalled_since>=.8:
                    self.cruising=False;self.moving_since=None;self.stalled_since=None
            else:self.stalled_since=None
        elif toward>=.35:
            if self.moving_since is None:self.moving_since=now
            if now-self.moving_since>=.12:
                self.cruising=True;self.stalled_since=None
        else:self.moving_since=None
        return sign*(.6 if self.cruising else 1.)


class StagingTranslationAssist:
    """Start a verified gait only after an actually delivered move stalls.

    This changes the requested speed, before the existing acceleration,
    full-body and measured-momentum checks. Predictive braking stays zero.
    """
    STALL_SECONDS=1.5
    PROGRESS_M=.01
    MIN_COMPONENT_MPS=.35
    MAX_SPEED_MPS=.50

    def __init__(self):
        self.anchor=None;self.stalled_since=None;self.active=False

    @classmethod
    def configuration(cls):
        return {'eligible_observation_profile':'unitree_go2_45',
                'requires_actual_policy_feedback':True,'stall_seconds':cls.STALL_SECONDS,
                'progress_m':cls.PROGRESS_M,'minimum_active_component_mps':cls.MIN_COMPONENT_MPS,
                'maximum_norm_mps':cls.MAX_SPEED_MPS,'existing_braking_and_collision_checks_retained':True}

    def speed(self,requested,body_direction,position,limiter,now):
        feedback=getattr(limiter,'policy_command',None)
        if feedback is None:
            self.anchor=None;self.stalled_since=None;self.active=False
            return requested
        position=np.asarray(position[:2],float)
        effective_request=(np.linalg.norm(np.asarray(feedback)[:2])>.10 and abs(float(feedback[2]))<.06)
        quiet=(np.linalg.norm(limiter.measured[:2])<.02 and abs(limiter.measured[2])<.06)
        if requested<=0. or not effective_request or not quiet:
            self.anchor=None;self.stalled_since=None
        elif self.anchor is None or np.linalg.norm(position-self.anchor)>self.PROGRESS_M:
            self.anchor=position.copy();self.stalled_since=float(now)
        elif now-self.stalled_since>=self.STALL_SECONDS:
            self.active=True
        component=float(np.max(np.abs(body_direction)))
        if self.active and requested>0. and component>1e-8:
            return min(self.MAX_SPEED_MPS,max(requested,self.MIN_COMPONENT_MPS/component))
        return requested


class LocalStagingControl:
    def __init__(self,position,heading,goal_heading,planner,boxes,now):
        self.origin=np.asarray(position[:2],float).copy();self.started=now
        self.goal_heading=heading+wrap(goal_heading-heading)
        self.selector=LocalTurnManeuver(self.goal_heading)
        self.target=self.selector._select_stage(planner,boxes,self.origin,heading,require_stepping_arc=True,require_stable_placement=True)
        self.phase='staging_move';self.stable_since=None;self.failed=self.target is None
        self.turn_response=TurnResponseAssist()
        self.yaw_rate=StagingYawRate()
        self.placement_latched=False
        self.translation_assist=StagingTranslationAssist()

    def update(self,position,heading,planner,boxes,limiter,now):
        position=np.asarray(position[:2],float)
        state={'local_staging':True,'target':position.tolist(),'target_heading_rad':self.goal_heading,
               'staging_phase':self.phase,'recovering_buffer':True,'needs_replan':False}
        if self.failed or now-self.started>20. or np.linalg.norm(position-self.origin)>1.2:
            self.failed=True
            return np.zeros(3),{**state,'needs_replan':True,'staging_phase':'staging_failed'}
        turn_end=heading+wrap(self.goal_heading-heading)
        # The selected turn pose already has a checked clearance margin.
        # Keep it for admission and target validation: silently dropping a
        # .05m selection to .025m can brake after a5mm grid-cell crossing,
        # before the gait has translated into the selected turning region.
        can_turn=body_rotation_clear(planner,boxes,position,heading,turn_end,margin=self.selector.stage_margin)
        placement_ready=can_turn and turn_placement_clear(planner,boxes,position,heading,turn_end,self.selector.stage_margin)
        arc_ready=can_turn and stepping_arc_clear(planner,boxes,position,heading,math.copysign(1.,turn_end-heading))
        # Enter from the interior, then keep permission while the actual
        # complete turn remains clear. Applying the interior threshold again
        # during braking only moves the chatter to a different boundary.
        if not can_turn:self.placement_latched=False
        elif placement_ready and arc_ready:self.placement_latched=True
        can_start_turn=self.placement_latched and can_turn and arc_ready
        state.update(turn_placement_radius_m=.5*planner.resolution,turn_placement_ready=bool(placement_ready),
                     turn_placement_latched=bool(self.placement_latched))
        settled=(np.linalg.norm(limiter.measured[:2])<.06 and abs(limiter.measured[2])<.10
                 and np.linalg.norm(limiter.command[:2])<.06 and abs(limiter.command[2])<.10)
        error=wrap(self.goal_heading-heading)
        if self.phase=='staging_move':
            if can_start_turn:
                self.phase='staging_brake';self.stable_since=None
            else:
                if (not turn_placement_clear(planner,boxes,self.target,heading,turn_end,self.selector.stage_margin)
                        or not stepping_arc_clear(planner,boxes,self.target,heading,math.copysign(1.,turn_end-heading))
                        or not body_translation_recovery_clear(planner,boxes,position,heading,self.target)):
                    self.target=self.selector._select_stage(planner,boxes,position,heading,require_stepping_arc=True,require_stable_placement=True)
                if self.target is None:
                    self.failed=True
                    return np.zeros(3),{**state,'needs_replan':True,'staging_phase':'staging_failed'}
                delta=self.target-position;distance=float(np.linalg.norm(delta))
                c,s=math.cos(heading),math.sin(heading)
                direction=delta/max(distance,1e-9)
                measured_world=np.array([c*limiter.measured[0]-s*limiter.measured[1],s*limiter.measured[0]+c*limiter.measured[1]])
                toward=float(np.dot(measured_world,direction))
                # Position/velocity feedback damps approach rather than
                # alternating fixed backward and sideways requests.
                # Sub-0.1 m/s requests can leave the rough policy standing
                # short of the safe turn region. Keep a stepping request,
                # then brake using measured travel, rather than asymptotically
                # requesting an ineffective tiny translation near its edge.
                body_direction=np.array([c*direction[0]+s*direction[1],-s*direction[0]+c*direction[1]])
                sideways=abs(body_direction[1])>abs(body_direction[0])
                minimum=.35 if sideways else .20
                maximum=.35 if sideways else .30
                stopping_distance=max(0.,toward)*(.65 if sideways else .30)+.015
                speed=(0. if toward>.06 and distance<stopping_distance else
                       float(np.clip(1.5*distance-.6*toward,minimum,maximum)))
                speed=self.translation_assist.speed(speed,body_direction,position,limiter,now)
                state.update(staging_translation_assist_active=self.translation_assist.active,
                             staging_requested_speed_mps=float(speed))
                world=direction*speed
                command=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],0.])
                return command,{**state,'target':self.target.tolist(),'staging_phase':self.phase}
        if self.phase in ['staging_brake','staging_turn_brake']:
            self.yaw_rate.reset()
            if settled:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.20:
                    if self.phase=='staging_turn_brake' and abs(error)<.12:return None
                    self.phase='staging_turn' if can_start_turn else 'staging_move'
                    self.stable_since=None
            else:self.stable_since=None
            return np.zeros(3),{**state,'staging_phase':self.phase}
        if self.phase=='staging_turn':
            if not can_turn and self.turn_response.assist_origin is None:
                self.phase='staging_brake';self.stable_since=None
                return np.zeros(3),{**state,'staging_phase':self.phase}
            stopping_angle=abs(limiter.measured[2])*.25+limiter.command[2]**2/(2*limiter.yaw_acceleration)+.03
            if (abs(error)<.08 or (error*limiter.measured[2]>0 and abs(limiter.measured[2])>.1 and abs(error)<stopping_angle)
                    or (error*limiter.measured[2]<0 and abs(limiter.measured[2])>.1)):
                self.phase='staging_turn_brake';self.stable_since=None
                return np.zeros(3),{**state,'staging_phase':self.phase}
            request=np.array([0.,0.,self.yaw_rate.request(error,limiter.measured[2],now)])
            # Keep the effective stepping gait until predictive braking.
            # Returning to pure yaw after only 0.18 rad repeatedly stalls it.
            request,detail=self.turn_response.update(request,position,heading,planner,boxes,limiter,now,maintain_arc=True)
            if detail.get('turn_assist')=='no_observed_safe_arc':
                self.phase='staging_move';self.turn_response=TurnResponseAssist()
                request=np.zeros(3)
            return request,{**state,'staging_phase':self.phase,**detail}
        return np.zeros(3),{**state,'staging_phase':self.phase}
