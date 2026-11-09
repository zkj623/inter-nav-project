"""Measured settling and continuous translation into a known safe turn pose."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver,turn_placement_clear,prepare_turn_plan
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear,body_translation_recovery_clear,body_motion_clear
from grutopia_extension.interactive_navigation.go2_turn_response import TurnResponseAssist,stepping_arc_clear


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


def native_translation_speed(requested,body_direction):
    """Effective finite-motion targets measured on the actual native policy.

    A recorded stalled state stayed still at -.35 and stepped at -.45 on
    the identical empty-floor plant. Preserve direction, the .50 norm cap,
    and explicit zero; the returned request still needs all motion checks.
    """
    if requested<=0.:return 0.
    component=float(np.max(np.abs(body_direction)))
    if component<1e-8:return 0.
    reverse=float(body_direction[0])<-.9*float(np.linalg.norm(body_direction))
    minimum=.45 if reverse else .35
    return min(.50,max(float(requested),minimum/component))


class TurnPositionHold:
    """Counter small uncommanded base drift while a yaw gait is already active."""
    def __init__(self):self.anchor=None

    def reset(self):self.anchor=None

    def apply(self,request,position,heading,measured):
        request=np.asarray(request,float).copy()
        if abs(request[2])<.10 or np.linalg.norm(request[:2])>.01:
            self.reset();return request
        position=np.asarray(position[:2],float)
        if self.anchor is None:self.anchor=position.copy()
        delta=self.anchor-position;c,s=math.cos(heading),math.sin(heading)
        error=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        correction=3.*error-.8*np.asarray(measured[:2],float)
        correction*=min(1.,.15/max(float(np.linalg.norm(correction)),1e-9))
        request[:2]=correction
        return request


class StagingYawRate:
    """Use the start request only until measured turning is established.

    The legacy rough policy cruises at .6 after its start request. The
    validated Unitree45 candidate keeps the configured 1 rad/s request
    until the enclosing measured braking decision.
    The enclosing limiter and body sweep still check the applied command.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sign=None;self.moving_since=None;self.stalled_since=None
        self.cruising=False

    def request(self,error,measured_yaw,now,*,native=False):
        sign=math.copysign(1.,error)
        if sign!=self.sign:
            self.reset();self.sign=sign
        # The .6 cruise schedule predates the Unitree45 policy. Keep its
        # already-supported 1 rad/s request continuous until the enclosing
        # measured braking decision, instead of repeatedly dropping a gait.
        if native:return sign
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
        self.start_anchor=None;self.start_moving_since=None;self.start_ready=False;self.starting=False

    @classmethod
    def configuration(cls):
        return {'eligible_observation_profile':'unitree_go2_45',
                'requires_actual_policy_feedback':True,'stall_seconds':cls.STALL_SECONDS,
                'progress_m':cls.PROGRESS_M,'minimum_active_component_mps':cls.MIN_COMPONENT_MPS,
                'minimum_near_reverse_component_mps':.45,'near_reverse_cosine':.9,
                'native_finite_motion_floor_until_explicit_brake':True,
                'maximum_norm_mps':cls.MAX_SPEED_MPS,'existing_braking_and_collision_checks_retained':True}

    def speed(self,requested,body_direction,position,limiter,now,*,effective_start=False):
        feedback=getattr(limiter,'policy_command',None)
        if feedback is None:
            self.anchor=None;self.stalled_since=None;self.active=False
            self.start_anchor=None;self.starting=False;self.start_ready=False;self.start_moving_since=None
            return requested
        position=np.asarray(position[:2],float)
        if requested<=0.:
            self.start_anchor=None;self.starting=False;self.start_ready=False;self.start_moving_since=None
        elif effective_start:
            if self.start_anchor is None:self.start_anchor=position.copy()
            toward=float(np.dot(limiter.measured[:2],body_direction))
            if toward>.06:
                if self.start_moving_since is None:self.start_moving_since=float(now)
                if now-self.start_moving_since>=.12 and np.linalg.norm(position-self.start_anchor)>.025:
                    self.start_ready=True
            else:self.start_moving_since=None
            # Native staging is a finite movement followed by an explicit
            # brake. Do not drop back into an ineffective .2 request merely
            # because the first few centimetres have executed.
            self.starting=True
        else:
            self.start_anchor=None;self.starting=False;self.start_ready=False;self.start_moving_since=None
        effective_request=(np.linalg.norm(np.asarray(feedback)[:2])>.10 and abs(float(feedback[2]))<.06)
        quiet=(np.linalg.norm(limiter.measured[:2])<.02 and abs(limiter.measured[2])<.06)
        if requested<=0. or not effective_request or not quiet:
            self.anchor=None;self.stalled_since=None
        elif self.anchor is None or np.linalg.norm(position-self.anchor)>self.PROGRESS_M:
            self.anchor=position.copy();self.stalled_since=float(now)
        elif now-self.stalled_since>=self.STALL_SECONDS:
            self.active=True
        component=float(np.max(np.abs(body_direction)))
        if (self.active or self.starting) and requested>0. and component>1e-8:
            return native_translation_speed(requested,body_direction)
        return requested


class TurnReversalGate:
    """Distinguish a gait oscillation from retained rotation away from the goal.

    A reverse-rate sample alone does not prove the commanded turn has
    reversed. Require both time and actual heading loss; a large excursion
    still brakes immediately. Geometry and measured-momentum checks remain
    independent of this gate.
    """
    def __init__(self):
        self.since=None;self.heading=None;self.sign=None

    def update(self,error,rate,heading,now):
        sign=math.copysign(1.,error)
        if sign!=self.sign:
            self.since=None;self.heading=None;self.sign=sign
        if sign*rate>=-.10:
            self.since=None;self.heading=None
            return False
        if self.since is None:
            self.since=float(now);self.heading=float(heading)
        loss=max(0.,-sign*wrap(heading-self.heading))
        return loss>.08 or (now-self.since>=.20 and loss>.03)


class LocalStagingControl:
    def __init__(self,position,heading,goal_heading,planner,boxes,now):
        self.origin=np.asarray(position[:2],float).copy();self.started=now
        self.goal_heading=heading+wrap(goal_heading-heading)
        self.selector=LocalTurnManeuver(self.goal_heading)
        self.target=self.selector._select_stage(planner,boxes,self.origin,heading,require_stepping_arc=True,require_stable_placement=True)
        self.escape_plan=None
        if self.target is None:
            self.escape_plan=prepare_turn_plan(planner,boxes,self.origin,heading,self.goal_heading)
            if self.escape_plan is not None:
                self.goal_heading=self.escape_plan['prepared_heading']
                self.selector=LocalTurnManeuver(self.goal_heading)
                self.selector.stage_margin=self.escape_plan['margin']
                self.target=self.origin.copy()
        self.phase='staging_move';self.stable_since=None;self.failed=self.target is None
        self.turn_response=TurnResponseAssist()
        self.yaw_rate=StagingYawRate()
        self.placement_latched=False
        self.translation_assist=StagingTranslationAssist()
        self.reversal_gate=TurnReversalGate()
        self.profile_initialized=self.escape_plan is not None
        self.translation_heading=None
        self.failure_reason='no_initial_turn_pose' if self.failed else None
        self.turn_position_hold=TurnPositionHold()

    def _finish_preparation(self,position,heading,planner,boxes):
        plan=self.escape_plan;goal=heading+wrap(plan['goal_heading']-heading)
        selector=LocalTurnManeuver(goal);selector.stage_margin=plan['margin'];target=plan['target'].copy()
        if not (body_translation_recovery_clear(planner,boxes,position,heading,target)
                and turn_placement_clear(planner,boxes,target,heading,goal,selector.stage_margin)
                and stepping_arc_clear(planner,boxes,target,heading,math.copysign(1.,goal-heading))):
            target=selector._select_stage(planner,boxes,position,heading,True,True)
        if target is None:
            self.failed=True;self.failure_reason='prepared_escape_handoff_blocked';return False
        self.goal_heading=goal;self.selector=selector;self.target=target;self.escape_plan=None
        self.phase='staging_move';self.stable_since=None;self.placement_latched=False
        self.translation_heading=None;self.translation_assist=StagingTranslationAssist()
        self.turn_response=TurnResponseAssist();self.yaw_rate.reset();self.reversal_gate=TurnReversalGate()
        self.turn_position_hold.reset()
        # Keep started and origin: preparation cannot renew time/distance budgets.
        return True

    def _select_target(self,planner,boxes,position,heading):
        maximum=self.selector.reserve_margin
        for reserve in sorted({maximum,*[r for r in [.05,.025,0.] if r<=maximum]},reverse=True):
            self.selector.reserve_margin=reserve
            target=self.selector._select_stage(planner,boxes,position,heading,
                require_stepping_arc=True,require_stable_placement=True)
            if target is not None:return target
        return None

    def update(self,position,heading,planner,boxes,limiter,now):
        position=np.asarray(position[:2],float)
        if not self.profile_initialized:
            self.profile_initialized=True
            if limiter.policy_command is not None and abs(wrap(self.goal_heading-heading))>.60:
                self.selector.reserve_margin=.08
                reserved=self._select_target(planner,boxes,position,heading)
                if reserved is not None:self.target=reserved
                else:self.selector.reserve_margin=0.
        state={'local_staging':True,'target':position.tolist(),'target_heading_rad':self.goal_heading,
               'staging_phase':self.phase,'recovering_buffer':True,'needs_replan':False,
               'turn_drift_reserve_m':self.selector.reserve_margin,'turn_clearance_margin_m':self.selector.stage_margin}
        if self.escape_plan is not None:
            state.update(preparatory_turn=True,final_heading_rad=self.escape_plan['goal_heading'],
                         prepared_translation_target=self.escape_plan['target'].tolist())
        if self.failed or now-self.started>20. or np.linalg.norm(position-self.origin)>1.2:
            self.failure_reason=self.failure_reason or ('stage_deadline' if now-self.started>20. else 'stage_translation_budget')
            self.failed=True
            return np.zeros(3),{**state,'needs_replan':True,'staging_phase':'staging_failed','staging_failure_reason':self.failure_reason}
        if self.phase!='staging_move':self.translation_heading=None
        if self.phase!='staging_turn':self.turn_position_hold.reset()
        turn_end=heading+wrap(self.goal_heading-heading)
        # The selected turn pose already has a checked clearance margin.
        # Keep it for admission and target validation: silently dropping a
        # .05m selection to .025m can brake after a5mm grid-cell crossing,
        # before the gait has translated into the selected turning region.
        can_turn=body_rotation_clear(planner,boxes,position,heading,turn_end,margin=self.selector.stage_margin)
        placement_margin=self.selector.stage_margin+self.selector.reserve_margin
        placement_ready=can_turn and turn_placement_clear(planner,boxes,position,heading,turn_end,placement_margin)
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
                if (not turn_placement_clear(planner,boxes,self.target,heading,turn_end,placement_margin)
                        or not stepping_arc_clear(planner,boxes,self.target,heading,math.copysign(1.,turn_end-heading))
                        or not body_translation_recovery_clear(planner,boxes,position,heading,self.target)):
                    self.target=self._select_target(planner,boxes,position,heading)
                    state['turn_drift_reserve_m']=self.selector.reserve_margin
                if self.target is None:
                    self.failed=True;self.failure_reason='no_reachable_turn_pose'
                    return np.zeros(3),{**state,'needs_replan':True,'staging_phase':'staging_failed','staging_failure_reason':self.failure_reason}
                if self.translation_heading is None:self.translation_heading=float(heading)
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
                speed=self.translation_assist.speed(speed,body_direction,position,limiter,now,
                    effective_start=limiter.policy_command is not None)
                state.update(staging_translation_assist_active=self.translation_assist.active,
                             staging_effective_start_active=self.translation_assist.starting,
                             staging_requested_speed_mps=float(speed))
                world=direction*speed
                yaw_hold=0.
                if limiter.policy_command is not None and body_motion_clear(planner,boxes,position,heading,0.,0.):
                    yaw_hold=float(np.clip(2.*wrap(self.translation_heading-heading)-.35*limiter.measured[2],-.25,.25))
                command=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],yaw_hold])
                state['staging_heading_hold_error_rad']=wrap(self.translation_heading-heading)
                return command,{**state,'target':self.target.tolist(),'staging_phase':self.phase}
        if self.phase in ['staging_brake','staging_turn_brake']:
            self.yaw_rate.reset()
            self.reversal_gate=TurnReversalGate()
            if settled:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.20:
                    if self.phase=='staging_turn_brake' and abs(error)<.12:
                        if self.escape_plan is None:return None
                        advanced=self._finish_preparation(position,heading,planner,boxes)
                        return np.zeros(3),{**state,'preparatory_turn_completed':advanced,
                            'target':self.target.tolist(),'target_heading_rad':self.goal_heading,
                            'staging_phase':self.phase if advanced else 'staging_failed','needs_replan':not advanced,
                            'staging_failure_reason':self.failure_reason}
                    self.phase='staging_turn' if can_start_turn else 'staging_move'
                    self.turn_response=TurnResponseAssist()
                    self.stable_since=None
            else:self.stable_since=None
            return np.zeros(3),{**state,'staging_phase':self.phase}
        if self.phase=='staging_turn':
            if not can_turn and self.turn_response.assist_origin is None:
                self.phase='staging_brake';self.stable_since=None
                return np.zeros(3),{**state,'staging_phase':self.phase,'turn_brake_reason':'actual_body_turn_sweep_blocked'}
            stopping_angle=abs(limiter.measured[2])*.25+limiter.command[2]**2/(2*limiter.yaw_acceleration)+.03
            reversed_motion=(error*limiter.measured[2]<0 and abs(limiter.measured[2])>.1)
            if limiter.policy_command is not None:
                reversed_motion=self.reversal_gate.update(error,limiter.measured[2],heading,now)
            if (abs(error)<.08 or (error*limiter.measured[2]>0 and abs(limiter.measured[2])>.1 and abs(error)<stopping_angle)
                    or reversed_motion):
                self.phase='staging_turn_brake';self.stable_since=None
                state['turn_brake_reason']='retained_reverse_motion' if reversed_motion else 'target_braking_distance'
                return np.zeros(3),{**state,'staging_phase':self.phase}
            native=limiter.policy_command is not None
            request=np.array([0.,0.,self.yaw_rate.request(error,limiter.measured[2],now,native=native)])
            state['turn_rate_profile']='unitree45_continuous' if native else 'legacy_measured_cruise'
            # Keep the effective stepping gait until predictive braking.
            # Returning to pure yaw after only 0.18 rad repeatedly stalls it.
            request,detail=self.turn_response.update(request,position,heading,planner,boxes,limiter,now,maintain_arc=True)
            if native:
                request=self.turn_position_hold.apply(request,position,heading,limiter.measured)
                state['turn_position_hold_command']=request[:2].tolist()
            if detail.get('turn_assist')=='no_observed_safe_arc':
                self.phase='staging_brake' if native else 'staging_move';self.stable_since=None;self.turn_response=TurnResponseAssist()
                request=np.zeros(3)
            return request,{**state,'staging_phase':self.phase,**detail}
        return np.zeros(3),{**state,'staging_phase':self.phase}
