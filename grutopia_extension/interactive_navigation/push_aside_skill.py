"""Parameterized Go2 side-push skill with staged alignment and map confirmation.

The skill emits velocity/navigation commands. It never sets robot or object
poses; object movement must come from the simulator's contact dynamics.
"""
from dataclasses import dataclass
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
import math
import numpy as np
from grutopia_extension.interactive_navigation.interaction_skill import SkillSpec,SkillCommand
from grutopia_extension.interactive_navigation.go2_contact_control import (
    ContactYawServo,contact_heading_hold,contact_face_lateral_allowance,observed_contact_face_clear,observed_contact_retreat,
    push_centerline_velocity,push_completion_state,push_braking_distance,observed_release_braking_clear)
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl,checked_contact_turn_command,checked_contact_step_command
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_rotation_clear,body_translation_recovery_clear,body_motion_clear,
    ForwardExplorationPathTracker,navigation_body_blocked,body_axis_recovery_target)
from grutopia_extension.interactive_navigation.go2_passage_confirmation import observed_passage_routes
from grutopia_extension.interactive_navigation.go2_exploration_tracking import decision_buffer_recovery
from grutopia_extension.interactive_navigation.smooth_go2_navigation import smooth_recovery_action
from grutopia_extension.interactive_navigation.go2_heading_trim import HeadingTrimControl
from grutopia_extension.interactive_navigation.go2_contact_translation_step import ContactTranslationStep


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


PREPARATION_SETTLING_STATES=frozenset({'settle_at_stage','settle_alignment','settle_recovered_preparation','settle_contact'})


@dataclass(frozen=True)
class PushAsideConfig:
    front_gap:float=.35
    front_contact_reach:float=.17  # 0.50 m front-foot envelope minus 0.33 m nose offset
    lateral_speed:float=.35
    forward_speed:float=.25
    reverse_speed:float=.35
    push_start_speed:float=.60
    push_max_speed:float=.95
    push_acceleration:float=.15
    max_push_braking_seconds:float=6.
    yaw_tolerance:float=.10
    lateral_tolerance:float=.055
    preparation_face_offset_max:float=.10
    pushing_face_offset_max:float=.15
    contact_face_margin:float=.04
    max_engage_standoff:float=.60
    alignment_tolerance:float=.035
    alignment_yaw_tolerance:float=.05
    stage_handoff_radius:float=.30
    preparation_heading_tolerance:float=.45
    max_preparation_recovery_seconds:float=12.
    max_retries:int=4
    max_skill_seconds:float=180.
    max_turn_seconds:float=40.
    max_alignment_seconds:float=20.
    max_verify_seconds:float=8.
    release_distance:float=.55


class PushAsideSkill:
    spec=SkillSpec('push_aside','1.0',('box',),'physically side-park a box and confirm an observed free passage')

    def __init__(self,permission,plan,task_start,task_goal,config=None):
        if permission is None or permission.skill_name!=self.spec.name or permission.category!='box':
            raise ValueError('PushAsideSkill requires an allowlisted box')
        if permission.source=='rgb_dino_qwen_vl':
            if len(set(permission.evidence_frames))<2 or not permission.classifier_models:
                raise ValueError('Vision mode requires two independent recognized frames')
        elif permission.source=='ground_truth_layout':
            if (not permission.source_reference or len(permission.source_digest)!=64
                    or any(c not in '0123456789abcdef' for c in permission.source_digest)
                    or permission.evidence_frames or permission.classifier_models):
                raise ValueError('Ground-truth mode requires explicit provenance without fake visual evidence')
        else:raise ValueError('Unsupported interaction permission source')
        if permission.object_id!=plan.box_id:raise ValueError('Permission does not match push target')
        self.permission=permission;self.plan=plan;self.config=config or PushAsideConfig()
        self.task_start=np.asarray(task_start,float);self.task_goal=np.asarray(task_goal,float)
        self.direction=np.asarray(plan.direction,float);self.direction/=np.linalg.norm(self.direction)
        self.tangent=np.array([-self.direction[1],self.direction[0]])
        self.push_heading=math.atan2(self.direction[1],self.direction[0])
        self.state='prepare';self.started=None;self.state_since=None;self.retries=0
        self.initial_box=None;self.last_box=None;self.engage_box_position=None;self.initial_box_geometry=None;self.last_box_motion_time=None;self.last_release_probe=-1e9;self.stage=None;self.anchor=None;self.axis=None
        self.turn=None;self.yaw_servo=ContactYawServo(self.config.yaw_tolerance)
        self.settled_since=None;self.translation_braking=False;self.navigation_goal=None;self.events=[]
        self.release_origin=None;self.release_reason=None;self.certificate=None
        self.last_verify=-1e9;self.observation_moves=0;self.used_stages=[]
        self.progress_mark=None;self.progress_since=None;self.max_tilt=0.
        self.terminal_release_reason=None;self.lateral_step=None
        self.reason='';self.status='running';self.recovery_from_contact=False;self.box_velocity=0.;self.last_box_time=None;self.push_brakes=0
        self.release_pose_recoveries=0;self.release_recovery_tracker=None;self.release_recovery_target=None
        self.release_recovery_diagnostic=None
        self.release_axis_attempts=[];self.release_axis_pending=False
        self.preparation_recoveries=0;self.recovery_tracker=None;self.recovery_target=None;self.next_stage_probe=-1e9;self.heading_trim=None

    def _transition(self,state,ctx,reason=''):
        self.events.append({'skill':self.spec.name,'object_id':self.plan.box_id,'from':self.state,
            'to':state,'reason':reason,'time_s':float(ctx.now),'retry':self.retries})
        self.state=state;self.state_since=ctx.now;self.settled_since=None;self.translation_braking=False;self.lateral_step=None
        if state=='recover_release_pose':self.release_axis_attempts=[];self.release_axis_pending=False
        if state=='refine_heading':self.heading_trim=HeadingTrimControl(self.push_heading,min(self.config.alignment_yaw_tolerance,self.config.yaw_tolerance),acceptance_tolerance=self.config.yaw_tolerance)
        self.progress_mark=np.asarray(ctx.position[:2]).copy();self.progress_since=ctx.now

    def _output(self,velocity=(0.,0.,0.),navigation_goal=None):
        events=self.events;self.events=[]
        return SkillCommand(self.state,tuple(map(float,velocity)),
            None if navigation_goal is None else tuple(map(float,navigation_goal)),
            self.status,self.reason,events,self.certificate)

    def _fail(self,ctx,reason):
        self.reason=reason;self.status='failed';self._transition('failed',ctx,reason)
        return self._output()

    def _quiet(self,ctx,seconds=.30,max_tilt=15.):
        commanded=np.asarray(getattr(ctx.limiter,'command',(0.,0.,0.)),float)
        angular=getattr(ctx,'body_angular_velocity',None)
        angular=np.asarray((0.,0.,ctx.yaw_rate) if angular is None else angular,float)
        quiet=(np.linalg.norm(ctx.body_velocity[:2])<.08 and abs(ctx.body_velocity[2])<.05
               and np.max(np.abs(angular))<.10 and ctx.tilt_degrees<max_tilt
               and np.linalg.norm(commanded[:2])<.03 and abs(commanded[2])<.06)
        if not quiet:self.settled_since=None;return False
        if self.settled_since is not None and (np.linalg.norm(np.asarray(ctx.position[:2])-self.settled_position[:2])>=.025
                or abs(float(ctx.position[2])-self.settled_position[2])>=.015
                or abs(ctx.tilt_degrees-self.settled_tilt)>=1.5):self.settled_since=None
        if self.settled_since is None:
            self.settled_since=ctx.now;self.settled_position=np.asarray(ctx.position).copy();self.settled_tilt=ctx.tilt_degrees
        return ctx.now-self.settled_since>=seconds

    def _box_frame(self,ctx):
        extent=float(np.dot(np.abs(self.direction),np.asarray(ctx.box.dimensions[:2]))/2)
        nose=np.asarray(ctx.position[:2])+.33*np.array([math.cos(ctx.heading),math.sin(ctx.heading)])
        delta=np.asarray(ctx.box.xy)-nose
        return float(np.dot(delta,self.direction)-extent),float(np.dot(delta,self.tangent)),wrap(self.push_heading-ctx.heading)

    def _contact_lateral_allowance(self,ctx,*,moving=False):
        maximum=(self.config.pushing_face_offset_max if moving else self.config.preparation_face_offset_max)
        return contact_face_lateral_allowance(ctx.box.dimensions,self.direction,
            wrap(self.push_heading-ctx.heading),ctx.tilt_degrees,maximum,self.config.contact_face_margin)

    def _contact_face_clear(self,ctx,*,moving=False):
        maximum=(self.config.pushing_face_offset_max if moving else self.config.preparation_face_offset_max)
        geometry=[getattr(ctx.box,k,None) for k in ['local_dimensions','orientation','center_z']]
        if any(value is not None for value in geometry):
            if any(value is None for value in geometry) or getattr(ctx,'orientation',None) is None:return False
            return observed_contact_face_clear(ctx.position,ctx.orientation,ctx.box.xy,geometry[2],
                geometry[0],geometry[1],self.direction,maximum,self.config.contact_face_margin)
        # Legacy static BoxState fixtures declare an axis-aligned face. The
        # live proposal source always supplies the observed rigid-body pose.
        if abs(float(getattr(ctx.box,'yaw',0.)))>1e-8:return False
        allowance=self._contact_lateral_allowance(ctx,moving=moving)
        return allowance>0 and abs(self._box_frame(ctx)[1])<allowance

    def _contact_ready(self,ctx):
        # Accept a verified approach corridor, not a tiny exact body pose.
        # The final task goal and measured-stop gates are unchanged.
        gap,lateral,error=self._box_frame(ctx)
        if (not self._contact_face_clear(ctx) or abs(error)>=self.config.yaw_tolerance
                or not .20<=gap<=self.config.max_engage_standoff):return False
        if not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):return False
        # Keep the selected box in this check. The endpoint is still outside
        # the front-foot contact envelope; intentional contact is checked by
        # the existing execution guard only after the measured stop.
        distance=max(0.,gap-(self.config.front_contact_reach+.075))
        target=np.asarray(ctx.position[:2])+self.direction*distance
        return body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,target)

    def _reuse_alignment_pose(self,ctx):
        """Keep a nearby, observed and already oriented retreat pose."""
        gap,lateral,yaw_error=self._box_frame(ctx)
        if any(np.linalg.norm(np.asarray(ctx.position[:2])-old)<.10 for old in self.used_stages):return False
        if abs(yaw_error)>self.config.preparation_heading_tolerance or not .20<=gap<=.85 or abs(lateral)>.80:return False
        if not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):return False
        extent=float(np.dot(np.abs(self.direction),np.asarray(ctx.box.dimensions[:2]))/2)
        anchor=np.asarray(ctx.box.xy)-self.direction*(extent+.33+self.config.front_gap)
        longitudinal=anchor+self.tangent*float(np.dot(np.asarray(ctx.position[:2])-anchor,self.tangent))
        # The turn controller executes the shortest yaw change. Near +/-pi,
        # raw endpoints would test almost a full revolution instead.
        end_heading=ctx.heading+wrap(self.push_heading-ctx.heading)
        if not body_rotation_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,end_heading,margin=.025):return False
        if not body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,self.push_heading,longitudinal):return False
        if not body_translation_recovery_clear(ctx.planner,ctx.boxes,longitudinal,self.push_heading,anchor):return False
        self.stage=np.asarray(ctx.position[:2]).copy();self.anchor=anchor
        passage=ctx.planner.active_passages.get(ctx.box.id,ctx.planner.passages.get(ctx.box.id))
        self.axis=np.asarray(passage['axis'] if passage else self.tangent,float)
        if np.dot(self.task_goal-np.asarray(ctx.box.xy),self.axis)<0:self.axis=-self.axis
        self.axis/=np.linalg.norm(self.axis)
        self.yaw_servo=ContactYawServo(min(self.config.alignment_yaw_tolerance,self.config.yaw_tolerance))
        return True

    def _select_stage(self,ctx):
        p=ctx.planner;position=np.asarray(ctx.position[:2])
        passage=p.active_passages.get(ctx.box.id,p.passages.get(ctx.box.id))
        self.axis=np.asarray(passage['axis'] if passage else self.tangent,float)
        if np.dot(self.task_goal-np.asarray(ctx.box.xy),self.axis)<0:self.axis=-self.axis
        self.axis/=np.linalg.norm(self.axis)
        extent=float(np.dot(np.abs(self.direction),np.asarray(ctx.box.dimensions[:2]))/2)
        with ctx.layers.global_planning() as boxes:
            heading=p.robot_heading;p.robot_heading=None
            try:
                candidates=[]
                for gap in [self.config.front_gap,.40,.30]:
                    anchor=np.asarray(ctx.box.xy)-self.direction*(extent+.33+gap)
                    if not body_rotation_clear(p,boxes,anchor,self.push_heading,self.push_heading,margin=.025):continue
                    points=[position,anchor,anchor-.25*self.direction,anchor-.45*self.direction]
                    points.extend(anchor-self.axis*back+self.direction*along
                                  for back in [.5,.7,.9,1.1] for along in [0.,.2,-.2,.4,-.4])
                    for point in points:
                        if any(np.linalg.norm(point-old)<.28 for old in self.used_stages):continue
                        if np.dot(np.asarray(ctx.box.xy)-point,self.direction)<extent+.5:continue
                        path=p.path(position,point,boxes) if np.linalg.norm(point-position)>.15 else [tuple(point)]
                        if not path:continue
                        arrival=ctx.heading if np.linalg.norm(point-position)<.15 else math.atan2(point[1]-path[-2][1],point[0]-path[-2][0]) if len(path)>1 else math.atan2(point[1]-position[1],point[0]-position[0])
                        end_heading=arrival+wrap(self.push_heading-arrival)
                        if not body_rotation_clear(p,boxes,point,arrival,end_heading,margin=.05):continue
                        longitudinal=anchor+self.tangent*float(np.dot(point-anchor,self.tangent))
                        if not body_translation_recovery_clear(p,boxes,point,self.push_heading,longitudinal):continue
                        if not body_translation_recovery_clear(p,boxes,longitudinal,self.push_heading,anchor):continue
                        lateral_travel=abs(float(np.dot(point-anchor,self.tangent)))
                        turn=abs(wrap(self.push_heading-arrival))
                        score=p.length(position,path)+.6*np.linalg.norm(point-anchor)+.6*lateral_travel+.25*turn
                        candidates.append((score,point.copy(),anchor.copy()))
                if not candidates:return False
                _,self.stage,self.anchor=min(candidates,key=lambda value:value[0])
                return True
            finally:p.robot_heading=heading

    def _retry(self,ctx,reason):
        if self.terminal_release_reason is not None:
            return self._fail(ctx,self.terminal_release_reason)
        recovery_source=self.state
        self.recovery_from_contact=recovery_source in {'engage','push','settle_push_progress','brake_after_push'}
        if reason=='heading_refinement_timeout' or reason.startswith('heading_trim_'):
            self.used_stages.append(np.asarray(ctx.position[:2]).copy())
        self.retries+=1;self.release_reason=reason
        self.release_origin=None
        if self.retries>self.config.max_retries:
            # A final partial push must still release contact and inspect the
            # actual passage. This grants no further preparation/push attempt
            # and uses exactly the existing fresh-map/full-body certificate.
            partial=(self.recovery_from_contact and self.engage_box_position is not None
                and self.initial_box_geometry is not None and self.last_box_motion_time is not None
                and ctx.box is not None and self.initial_box is not None
                and float(np.dot(np.asarray(ctx.box.xy)-self.initial_box,self.direction))>.25)
            if not partial:return self._fail(ctx,'retry_limit:'+reason)
            self.terminal_release_reason='retry_limit:'+reason
        self._transition('brake_for_recovery',ctx,reason)
        self.events[-1].update(recovery_source_state=recovery_source,recovery_from_contact=self.recovery_from_contact,
            terminal_release=self.terminal_release_reason is not None)
        return self._output()

    def _start_corrective_lateral_step(self,ctx):
        # The runner supplies policy_command only for a loader-validated 45D
        # Unitree profile. Legacy profiles keep their established behavior.
        error=float(np.dot(np.asarray(ctx.box.xy)-np.asarray(ctx.position[:2]),self.tangent))
        if (getattr(ctx,'policy_command',None) is None or
                abs(self.config.lateral_speed-ContactTranslationStep.SPEED)>1e-6 or
                abs(error)<=self.config.alignment_tolerance):return
        required=min(ContactTranslationStep.TARGET_PROGRESS_CAP,max(0.,abs(error)-self.config.alignment_tolerance))
        self.lateral_step=ContactTranslationStep(ctx.position,self.tangent*np.sign(error),ctx.now,minimum_progress_m=required)
        self.events[-1]['corrective_lateral_step']={**ContactTranslationStep.configuration(),'minimum_progress_m':required}

    def _translation(self,ctx,target,axis,speed,tolerance):
        delta=np.asarray(target)-np.asarray(ctx.position[:2]);error=float(np.dot(delta,axis))
        if self.translation_braking:
            if not self._quiet(ctx):return np.zeros(3),False
            self.translation_braking=False
            # A measured brake that repeatedly leaves the initial centering
            # short needs the same bounded input as a failed contact settle.
            # Log a control adjustment without resetting the phase deadline.
            if (self.state=='align_lateral' and self.lateral_step is None
                    and getattr(ctx,'policy_command',None) is not None
                    and abs(self.config.lateral_speed-ContactTranslationStep.SPEED)<1e-6
                    and abs(error)>tolerance and self.progress_since is not None
                    and ctx.now-self.progress_since>=ContactTranslationStep.PROGRESS_SECONDS):
                self.events.append({'skill':self.spec.name,'object_id':self.plan.box_id,
                    'from':self.state,'to':self.state,'reason':'settled_initial_lateral_plateau',
                    'time_s':float(ctx.now),'retry':self.retries,'lateral_error_m':abs(error),
                    'progress_stall_seconds':float(ctx.now-self.progress_since),
                    'alignment_deadline_preserved':True})
                self._start_corrective_lateral_step(ctx)
            return np.zeros(3),abs(error)<tolerance
        if abs(error)<tolerance:
            self.translation_braking=True
            return np.zeros(3),self._quiet(ctx)
        self.settled_since=None
        c,s=math.cos(ctx.heading),math.sin(ctx.heading);rot=np.array([[c,-s],[s,c]])
        world_v=rot@np.asarray(ctx.body_velocity[:2]);unit=axis*math.copysign(1.,error)
        toward=float(np.dot(world_v,unit));lateral=abs(np.dot(axis,self.tangent))>.5
        commanded=np.asarray(getattr(ctx.limiter,'command',(0.,0.,0.)),float)
        command_toward=max(0.,float(np.dot(rot@commanded[:2],unit)))
        acceleration=getattr(ctx.limiter,'linear_acceleration',.65)
        stopping_distance=.025+max(0.,toward)*(.65 if lateral else .4)+command_toward**2/(2*acceleration)
        if toward>.06 and abs(error)<stopping_distance:
            # Finish one deceleration before requesting another stride. Without
            # this latch, falling measured speed immediately restarted the gait.
            self.translation_braking=True
            return np.zeros(3),False
        world=unit*speed
        yaw_request=contact_heading_hold(wrap(self.push_heading-ctx.heading),ctx.yaw_rate)
        if not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):yaw_request=0.
        request=[*(rot.T@world),yaw_request]
        command,_=checked_contact_turn_command(request,ctx.position,ctx.heading,ctx.planner,ctx.boxes,
            ctx.limiter,ctx.now,target_position=np.asarray(ctx.position[:2])+axis*error)
        return command,False

    def _observed_passage_effect(self,ctx,along,commit=False):
        """Use the observed effect after a real withdrawal, not a looser distance.

        A current global route, the complete local body corridor, and the
        original box's counterfactual obstruction must all be established.
        This never certifies reaching the navigation goal; the caller still
        executes the corridor and checks the final stopped robot pose.
        """
        layers=ctx.layers
        if (along<=.25 or layers is None or self.engage_box_position is None or self.initial_box_geometry is None
                or self.last_box_motion_time is None
                or getattr(layers,'last_local',-1.)<=self.last_box_motion_time
                or self.plan.box_id not in getattr(layers,'visible_box_ids',())):
            return False
        tracked=next((box for box in layers.current_boxes if box.id==self.plan.box_id),None)
        if tracked is None:return False
        planner=layers.planner
        if not body_motion_clear(planner,layers.current_boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):return False
        confirmation_stage=self.state
        original_passage=planner.passages.get(self.plan.box_id)
        original_active=planner.active_passages.get(self.plan.box_id)
        had_hint=hasattr(layers,'passage_observation_hint');original_hint=getattr(layers,'passage_observation_hint',None)
        accepted=False
        try:
            layers.refresh_global(reason='recovered_push_effect_observation')
            proof,route,full=observed_passage_routes(layers,tracked,self.task_start,self.task_goal,
                np.asarray(ctx.position[:2]),layers.current_boxes)
            if not (proof and route and full):return False
            heading=math.atan2(proof['axis'][1],proof['axis'][0])
            before=[box for box in layers.current_boxes if box.id!=self.plan.box_id]+[self.initial_box_geometry]
            if body_rotation_clear(planner,before,proof['center'],heading,heading,margin=proof['body_margin_m']):
                return False  # A distant detour is not an effect of this push.
            accepted=True
            return {**proof,'confirmation_stage':confirmation_stage,
                'original_box_would_block_corridor':True,'original_box':vars(self.initial_box_geometry),
                'original_box_xy':list(self.initial_box_geometry.xy),
                'actual_push_distance_m':along,'nominal_planned_push_m':self.plan.push_distance,
                'confirming_observation_time_s':float(layers.last_local),
                'last_box_motion_time_s':float(self.last_box_motion_time)}
        finally:
            if not accepted or not commit:
                if original_passage is None:planner.passages.pop(self.plan.box_id,None)
                else:planner.passages[self.plan.box_id]=original_passage
                if original_active is None:planner.active_passages.pop(self.plan.box_id,None)
                else:planner.active_passages[self.plan.box_id]=original_active
                if had_hint:layers.passage_observation_hint=original_hint
                elif hasattr(layers,'passage_observation_hint'):del layers.passage_observation_hint

    def _recovered_passage_effect(self,ctx,along):
        # A recovered nominal footprint must retain the same observed
        # braking reserve required by the direct withdrawal path.
        if (getattr(ctx,'policy_command',None) is not None and
                not observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)):
            return False
        proof=self._observed_passage_effect(ctx,along,commit=True)
        if not proof:return False
        self.certificate=proof;self.status='succeeded';self.reason='observed_passage_created'
        self._transition('succeeded',ctx,self.reason)
        self.events[-1]['confirmation_stage']=proof['confirmation_stage']
        return True

    def _centering_state_after_stop(self,ctx):
        # The caller already established a measured stop. Reuse it when the
        # actual standoff is safe instead of adding another stationary dwell.
        gap,_,_=self._box_frame(ctx)
        safe=(.20<=gap<=self.config.max_engage_standoff and
            body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.))
        return 'align_lateral' if safe else 'align_longitudinal'

    def _begin_release_pose_recovery(self,ctx,along,*,blocked_contact=False):
        # Restore the actual body pose after a real push. This can be needed
        # before the old straight withdrawal has cleared the front feet, as
        # well as after braking out of a previously observed free corridor.
        # Reuse the original full-body navigation recovery with ALL boxes and
        # unknown cells; the target box is never excluded from these checks.
        gap,_,_=self._box_frame(ctx)
        if blocked_contact:
            layers=ctx.layers
            if (layers is None or self.plan.box_id not in getattr(layers,'visible_box_ids',())
                    or self.last_box_motion_time is None
                    or getattr(layers,'last_local',-1.)<=self.last_box_motion_time
                    or ctx.now-layers.last_local>2*float(getattr(layers,'local_period',.1))):
                return False
        # Escaping a blocked contact pose must not require a successful push.
        # Fresh visible geometry, a measured stop in the caller, and the full
        # checked recovery remain mandatory. The ordinary post-push release
        # and every passage-effect certificate still require actual progress.
        if ((not blocked_contact and along<=.25) or self.engage_box_position is None
                or (not blocked_contact and gap<self.config.front_contact_reach+.02)
                or self.release_pose_recoveries>=self.config.max_retries
                or body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)):
            return False
        tracker=ForwardExplorationPathTracker(ctx.position[:2],[tuple(self.task_goal)],guard_motion=False)
        action,tracking=tracker.action(ctx.position[:2],ctx.heading,ctx.planner,ctx.boxes)
        if tracking.get('needs_replan') or not tracking.get('recovering_buffer'):return False
        if not np.any(np.asarray(action['move_by_speed'],float)):return False
        self.release_recovery_tracker=tracker
        self.release_recovery_target=np.asarray(tracking['target'][:2])
        self.release_pose_recoveries+=1
        self._transition('recover_release_pose',ctx,'blocked_contact_withdrawal' if blocked_contact else 'settled_release_body_overlap')
        self.events[-1].update(recovery_target=self.release_recovery_target.tolist(),
            release_pose_recovery=self.release_pose_recoveries)
        return True

    def step(self,ctx):
        if self.status!='running':return self._output()
        if ctx.box is None or ctx.box.id!=self.plan.box_id:return self._fail(ctx,'target_track_lost')
        if self.started is None:
            self.started=ctx.now;self.state_since=ctx.now;self.initial_box=np.asarray(ctx.box.xy).copy()
            if hasattr(ctx.box,'dimensions'):
                self.initial_box_geometry=BoxState(ctx.box.id,tuple(ctx.box.xy),tuple(ctx.box.dimensions),
                    getattr(ctx.box,'yaw',0.),getattr(ctx.box,'mass_kg',1.))
        self.max_tilt=max(self.max_tilt,float(ctx.tilt_degrees))
        if not ctx.healthy or ctx.tilt_degrees>=30:return self._fail(ctx,'robot_unstable')
        if ctx.now-self.started>self.config.max_skill_seconds:return self._fail(ctx,'skill_timeout')
        if self.state in PREPARATION_SETTLING_STATES:
            if ctx.now-self.state_since>self.config.max_alignment_seconds:
                reason=('alignment_settling_timeout' if self.state=='settle_alignment' else
                        'alignment_timeout' if self.state=='settle_contact' else 'preparation_settling_timeout')
                self.used_stages.append(np.asarray(ctx.position[:2]).copy())
                return self._retry(ctx,reason)
            if ctx.tilt_degrees>=18.:
                # This phase already requests a stop. Allow its measured
                # settling gate to finish; do not spend a new retry merely
                # because braking briefly crosses the movement stop trigger.
                # The30degree hard limit above and15degree quiet gate remain.
                self.settled_since=None
                return self._output()
        elif self.state in ['turn_at_stage','refine_heading','recover_preparation','align_longitudinal','align_lateral'] and ctx.tilt_degrees>=18.:
            self.used_stages.append(np.asarray(ctx.position[:2]).copy())
            return self._retry(ctx,'preparation_attitude')
        moved=np.asarray(ctx.box.xy)-self.initial_box;along=float(np.dot(moved,self.direction))
        cross=float(np.dot(moved,self.tangent))
        if self.last_box is not None and ctx.now>self.last_box_time:
            if np.linalg.norm(np.asarray(ctx.box.xy)-self.last_box)>1e-5:self.last_box_motion_time=ctx.now
            observed=float(np.dot(np.asarray(ctx.box.xy)-self.last_box,self.direction))/(ctx.now-self.last_box_time)
            self.box_velocity=.8*self.box_velocity+.2*observed
        self.last_box=np.asarray(ctx.box.xy).copy();self.last_box_time=ctx.now
        if abs(cross)>.15:return self._fail(ctx,'box_left_safe_push_line')
        gap,lateral,yaw_error=self._box_frame(ctx)

        if self.state=='prepare':
            if self.terminal_release_reason is not None:return self._fail(ctx,self.terminal_release_reason)
            if self._reuse_alignment_pose(ctx):
                next_state='settle_at_stage' if abs(yaw_error)>self.config.yaw_tolerance else 'settle_alignment'
                self._transition(next_state,ctx,'reuse_observed_aligned_pose')
                return self._output()
            if not self._select_stage(ctx):
                if self.preparation_recoveries>=self.config.max_retries:return self._fail(ctx,'no_observed_preparation_pose')
                self.preparation_recoveries+=1
                extent=float(np.dot(np.abs(self.direction),np.asarray(ctx.box.dimensions[:2]))/2)
                self.recovery_waypoint=np.asarray(ctx.box.xy)-self.direction*(extent+.33+self.config.front_gap)
                self.recovery_tracker=ForwardExplorationPathTracker(ctx.position[:2],
                    [tuple(self.recovery_waypoint)],guard_motion=False)
                self.next_stage_probe=ctx.now+.5
                self._transition('recover_preparation',ctx,'leave_observed_body_or_global_buffer')
                return self._output()
            self.navigation_goal=self.stage;self._transition('navigate_to_stage',ctx)
            return self._output(navigation_goal=self.stage)
        if self.state=='recover_preparation':
            if ctx.now-self.state_since>self.config.max_preparation_recovery_seconds:
                return self._fail(ctx,'no_observed_preparation_after_recovery')
            if ctx.now>=self.next_stage_probe:
                self.next_stage_probe=ctx.now+.5
                if self._reuse_alignment_pose(ctx) or self._select_stage(ctx):
                    self._transition('settle_recovered_preparation',ctx,'observed_preparation_access_restored')
                    return self._output()
            recovery=None
            blocked=navigation_body_blocked(ctx.planner,ctx.boxes,ctx.heading)
            cell=ctx.planner.cell(ctx.position)
            if ctx.planner.valid_cell(cell) and blocked[cell]:
                recovery=self.recovery_tracker.action(ctx.position[:2],ctx.heading,ctx.planner,ctx.boxes)
                if recovery[1].get('needs_replan') or not recovery[1].get('recovering_buffer'):recovery=None
            if recovery is None:
                local_free=ctx.planner.free.copy()
                with ctx.layers.global_planning() as global_boxes:
                    recovery=decision_buffer_recovery(ctx.position[:2],ctx.heading,ctx.planner,
                        global_boxes,self.recovery_waypoint,ctx.heading,local_free=local_free,
                        local_boxes=ctx.boxes,require_goal_connection=False)
            if recovery is None:return self._output()
            action,tracking=recovery;self.recovery_target=np.asarray(tracking['target'][:2])
            if ctx.limiter is not None:
                action=smooth_recovery_action(action,ctx.position[:2],ctx.heading,ctx.planner,
                    ctx.boxes,ctx.limiter,ctx.now)
            return self._output(action['move_by_speed'])
        if self.state=='settle_recovered_preparation':
            if self._quiet(ctx):
                if ctx.limiter is not None:ctx.limiter.clear_guidance(ctx.position,ctx.now)
                ctx.layers.refresh_global(reason='preparation_buffer_recovered')
                self.recovery_tracker=None;self.recovery_target=None
                self._transition('prepare',ctx,'replan_from_settled_recovery_pose')
            return self._output()
        if self.state=='navigate_to_stage':
            if not ctx.navigation_reached:
                # A stage is a safe turning region, not the final task goal.
                # Accept the measured nearby pose only after checking the full
                # intended turn here; then brake and settle before turning.
                nearby=np.linalg.norm(np.asarray(ctx.position[:2])-self.stage)<self.config.stage_handoff_radius
                end_heading=ctx.heading+wrap(self.push_heading-ctx.heading)
                rejected=any(np.linalg.norm(np.asarray(ctx.position[:2])-old)<.10 for old in self.used_stages)
                if nearby and not rejected and body_rotation_clear(ctx.planner,ctx.boxes,ctx.position,
                        ctx.heading,end_heading,margin=.05):
                    self.stage=np.asarray(ctx.position[:2]).copy()
                    self._transition('settle_at_stage',ctx,'actual_safe_stage_pose')
                    return self._output()
                return self._output(navigation_goal=self.stage)
            self._transition('settle_at_stage',ctx);return self._output()
        if self.state=='settle_at_stage':
            if self._quiet(ctx):
                if self._recovered_passage_effect(ctx,along):return self._output()
                self.turn=ContactTurnControl(ctx.position,ctx.heading,self.push_heading,ctx.planner,ctx.boxes,ctx.now)
                self._transition('turn_at_stage',ctx)
            return self._output()
        if self.state=='turn_at_stage':
            result=self.turn.update(ctx.position,ctx.heading,ctx.planner,ctx.boxes,ctx.limiter,ctx.now)
            if self.turn.failed:
                if ctx.now-self.state_since<self.config.max_turn_seconds and abs(yaw_error)>.10:
                    self.turn=ContactTurnControl(ctx.position,ctx.heading,self.push_heading,ctx.planner,ctx.boxes,ctx.now)
                    return self._output()
                self.used_stages.append(np.asarray(ctx.position[:2]).copy())
                return self._retry(ctx,'turn_preparation_failed')
            if result is None:
                # Centering already holds heading while the dog is stepping.
                # Enforce the contact tolerance at the final contact pose;
                # do not demand a finer stationary yaw at every coarse turn.
                # ContactTurnControl has already completed its measured
                # translation/yaw stop. If standoff is safe, reuse that gate
                # for centering instead of inserting a second idle dwell.
                # In T05 v84 the rough policy resumed yaw drift during that
                # duplicate dwell, exhausting retries before any sidestep.
                next_state=self._centering_state_after_stop(ctx)
                safe_standoff=next_state=='align_lateral'
                self._transition(next_state,ctx,'coarse_heading_ready_for_centering')
                self.events[-1]['reused_turn_stop_for_standoff']=bool(safe_standoff)
                return self._output()
            return self._output(result[0])
        if self.state=='settle_alignment':
            if ctx.now-self.state_since>self.config.max_alignment_seconds:return self._retry(ctx,'alignment_settling_timeout')
            if self._quiet(ctx):
                if self._recovered_passage_effect(ctx,along):return self._output()
                if abs(yaw_error)>self.config.yaw_tolerance:
                    self.turn=ContactTurnControl(ctx.position,ctx.heading,self.push_heading,ctx.planner,ctx.boxes,ctx.now)
                    self._transition('turn_at_stage',ctx)
                else:self._transition(self._centering_state_after_stop(ctx),ctx,'settled_heading_ready_for_centering')
            return self._output()
        if self.state=='refine_heading':
            if ctx.now-self.state_since>self.config.max_alignment_seconds:return self._retry(ctx,'heading_refinement_timeout')
            if self.heading_trim is None:
                self.heading_trim=HeadingTrimControl(self.push_heading,min(self.config.alignment_yaw_tolerance,self.config.yaw_tolerance),acceptance_tolerance=self.config.yaw_tolerance)
            angular=getattr(ctx,'body_angular_velocity',None)
            angular=np.asarray((0.,0.,ctx.yaw_rate) if angular is None else angular,float)
            command,done,_=self.heading_trim.update(ctx.position,ctx.heading,ctx.planner,ctx.boxes,
                ctx.limiter,ctx.now,ctx.body_velocity,angular,ctx.tilt_degrees,
                policy_command=getattr(ctx,'policy_command',None))
            if self.heading_trim.failed:return self._retry(ctx,self.heading_trim.failed)
            if done:self._transition(self._centering_state_after_stop(ctx),ctx)
            return self._output(command)
        if self.state in ['align_longitudinal','align_lateral','settle_contact']:
            if abs(yaw_error)>.18:return self._retry(ctx,'heading_drift_during_alignment')
            if ctx.now-self.state_since>self.config.max_alignment_seconds:return self._retry(ctx,'alignment_timeout')
            actual_ready=self._contact_ready(ctx)
            if self.state!='settle_contact' and actual_ready and self.lateral_step is None:
                self._transition('settle_contact',ctx,'actual_contact_pose_ready')
                self.events[-1].update(nose_gap_m=gap,nose_lateral_error_m=lateral,
                    contact_face_center_offset_cap_m=self.config.preparation_face_offset_max,
                    contact_face_margin_m=self.config.contact_face_margin)
                return self._output()
            if self.state=='align_longitudinal':
                # Use the same measured nose standoff interval as final contact
                # validation. Do not chase a nominal anchor inside a safe band.
                if .20<=gap<=self.config.max_engage_standoff and body_motion_clear(
                        ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):
                    if self._quiet(ctx):self._transition('align_lateral',ctx,'safe_standoff_reached')
                    return self._output()
                target=self.anchor+self.tangent*float(np.dot(np.asarray(ctx.position[:2])-self.anchor,self.tangent))
                error=float(np.dot(target-np.asarray(ctx.position[:2]),self.direction))
                command,done=self._translation(ctx,target,self.direction,
                    self.config.forward_speed if error>=0 else self.config.reverse_speed,.05)
                if done:self._transition('align_lateral',ctx)
                return self._output(command)
            if self.state=='align_lateral':
                # A long sidestep may drift longitudinally. Brake and correct
                # the standoff before resuming lateral work outside its band.
                if gap<.20 or gap>self.config.max_engage_standoff+.05:
                    self._transition('settle_alignment',ctx,'standoff_drift_during_sidestep')
                    return self._output()
                # The validated anchor is a BODY pose at push_heading.
                # Centering the currently rotated nose shifts that target by
                # .33*sin(yaw_error), then final heading trim undoes it. Track
                # the same body centerline that preparation planning checked;
                # final contact still validates the actual nose and heading.
                body_lateral=float(np.dot(np.asarray(ctx.box.xy)-np.asarray(ctx.position[:2]),self.tangent))
                body_target=np.asarray(ctx.position[:2])+self.tangent*body_lateral
                if (self.progress_mark is None or self.progress_since is None or
                        abs(float(np.dot(np.asarray(ctx.position[:2])-self.progress_mark,self.tangent)))>=ContactTranslationStep.PROGRESS_DISTANCE):
                    self.progress_mark=np.asarray(ctx.position[:2]).copy();self.progress_since=ctx.now
                if self.lateral_step is not None:
                    # The outer face band can be entered by a temporary body
                    # lean. Finish bounded input delivery or reach the inner
                    # centerline band, then re-evaluate the original quiet gate.
                    if abs(body_lateral)<self.config.alignment_tolerance:
                        finite_used=self.lateral_step.finite_target_check_used
                        self._transition('settle_contact',ctx,'lateral_inner_alignment_observed')
                        self.events[-1]['finite_target_check_used']=finite_used
                        return self._output()
                    world,detail=self.lateral_step.update(ctx.position,ctx.heading,
                        getattr(ctx,'policy_command',None),ctx.now)
                    if self.lateral_step.failed:return self._retry(ctx,self.lateral_step.failed)
                    if self.lateral_step.complete:
                        self._transition('settle_contact',ctx,'lateral_measured_step_complete')
                        self.events[-1]['lateral_step']=detail
                        return self._output()
                    c,s=math.cos(ctx.heading),math.sin(ctx.heading);rot=np.array([[c,-s],[s,c]])
                    # Keep this bounded correction translational. Actual yaw
                    # momentum is still checked, and the original stopped
                    # heading/face gates decide whether contact may begin.
                    request=[*(rot.T@world),0.]
                    command,_,mode=checked_contact_step_command(request,ctx.position,ctx.heading,
                        ctx.planner,ctx.boxes,ctx.limiter,ctx.now,target_position=body_target)
                    self.lateral_step.last_motion_check=mode
                    self.lateral_step.finite_target_check_used |= mode=='finite_target'
                    return self._output(command)
                command,done=self._translation(ctx,body_target,self.tangent,self.config.lateral_speed,self.config.alignment_tolerance)
                if done:self._transition('settle_contact',ctx)
                return self._output(command)
            ready=self._contact_ready(ctx)
            # Geometry observed while coasting is not a settled preparation
            # pose. Finish braking before selecting any heading/translation
            # correction, so a trim does not spend its travel budget on drift.
            if not self._quiet(ctx,.4):return self._output()
            if ready:
                if self._recovered_passage_effect(ctx,along):return self._output()
                self.engage_box_position=np.asarray(ctx.box.xy).copy()
                self._transition('engage',ctx);return self._output()
            if gap<.20 or gap>self.config.max_engage_standoff+.05:
                self._transition('align_longitudinal',ctx,'settled_standoff_outside_preparation_band')
                return self._output()
            body_lateral=abs(float(np.dot(np.asarray(ctx.box.xy)-np.asarray(ctx.position[:2]),self.tangent)))
            if body_lateral>self.config.preparation_face_offset_max:
                # Correct a large sidestep overshoot before trying a fine
                # turn next to the jamb. The actual 3D face gate still decides
                # when deliberate contact is permitted.
                self._transition('align_lateral',ctx,'settled_lateral_overshoot')
                self._start_corrective_lateral_step(ctx)
                return self._output()
            if abs(yaw_error)>self.config.yaw_tolerance:
                self._transition('refine_heading',ctx,'contact_heading_outside_tolerance')
                return self._output()
            if not self._contact_face_clear(ctx):
                self._transition('align_lateral',ctx)
                self._start_corrective_lateral_step(ctx)
            elif gap>self.config.max_engage_standoff or gap<.20:self._transition('align_longitudinal',ctx)
            return self._output()
        if self.state in ['engage','push','settle_push_progress','brake_after_push'] and along>self.plan.push_distance+.25:
            return self._fail(ctx,'safe_push_distance_exceeded')
        if self.state=='settle_push_progress':
            if self._quiet(ctx,.4,max_tilt=18.):
                clearance=ctx.planner.clearance(ctx.box)
                if push_completion_state(None,clearance,along,True) and along>=self.plan.push_distance-.035:
                    self.release_origin=np.asarray(ctx.position[:2]).copy();self.release_reason='side_parked'
                    self._transition('retreat',ctx,'side_parked_after_measured_braking')
                else:
                    # A predicted stop is not a clearance certificate. Keep
                    # the original required displacement and finish shortfalls.
                    self.engage_box_position=np.asarray(ctx.box.xy).copy()
                    self._transition('engage',ctx,'shortfall_after_measured_braking')
                return self._output()
            if ctx.now-self.state_since>self.config.max_push_braking_seconds:
                return self._retry(ctx,'push_braking_not_settled')
            return self._output()
        if self.state in ['engage','push']:
            if ctx.tilt_degrees>18:return self._retry(ctx,'contact_attitude')
            if abs(yaw_error)>.18 or not self._contact_face_clear(ctx,moving=True):return self._retry(ctx,'contact_alignment_drift')
            if self.state=='engage' and (gap<=self.config.front_contact_reach+.02 or float(np.dot(np.asarray(ctx.box.xy)-self.engage_box_position,self.direction))>.015):
                self._transition('push',ctx);self.push_progress=along;self.push_progress_time=ctx.now
            clearance=ctx.planner.clearance(ctx.box)
            done=push_completion_state(None,clearance,along,True)
            if done and along>=self.plan.push_distance-.035:
                self.release_origin=np.asarray(ctx.position[:2]).copy();self.release_reason='side_parked'
                self._transition('brake_after_push',ctx,done);return self._output()
            c,s=math.cos(ctx.heading),math.sin(ctx.heading);rot=np.array([[c,-s],[s,c]])
            measured=np.asarray(getattr(ctx.limiter,'measured',ctx.body_velocity),float)
            push_velocity=float(np.dot(rot@measured[:2],self.direction))
            attempt_progress=(0. if self.engage_box_position is None else
                float(np.dot(np.asarray(ctx.box.xy)-self.engage_box_position,self.direction)))
            stopping_distance=push_braking_distance(push_velocity)
            # After a partial push, the robot is accelerating toward contact
            # again. Historical box displacement cannot prove that THIS push
            # is moving the box or will coast with the approaching robot.
            if (self.state=='push' and along>.10 and attempt_progress>.015
                    and self.plan.push_distance-along<=stopping_distance):
                self.push_brakes+=1
                self._transition('settle_push_progress',ctx,'predicted_push_coast')
                self.events[-1].update(actual_push_m=along,attempt_push_m=attempt_progress,
                    remaining_push_m=self.plan.push_distance-along,
                    filtered_push_speed_mps=push_velocity,predicted_coast_m=stopping_distance)
                return self._output()
            elapsed=ctx.now-self.state_since
            if self.state=='engage' and elapsed>12:return self._retry(ctx,'contact_not_established')
            if self.state=='push':
                if along-self.push_progress>.015:self.push_progress=along;self.push_progress_time=ctx.now
                if ctx.now-self.push_progress_time>6:return self._retry(ctx,'box_motion_stalled')
            speed=min(self.config.push_start_speed,self.config.push_max_speed) if self.state=='engage' else min(self.config.push_max_speed,self.config.push_start_speed+self.config.push_acceleration*elapsed)
            c,s=math.cos(ctx.heading),math.sin(ctx.heading);rot=np.array([[c,-s],[s,c]])
            world_v=rot@np.asarray(ctx.body_velocity[:2])
            velocity=push_centerline_velocity(self.direction,speed,lateral,float(np.dot(world_v,self.tangent)))
            rate=float(np.clip(3.*yaw_error-.35*ctx.yaw_rate,-.55,.55))
            request=np.array([*(rot.T@velocity),rate]);command=ctx.limiter.preview(request,ctx.now)
            return self._output(command)
        if self.state in ['brake_for_recovery','brake_after_push']:
            if self._quiet(ctx,.4,max_tilt=18.) or ctx.now-self.state_since>1.5:
                if self.release_origin is None:self.release_origin=np.asarray(ctx.position[:2]).copy()
                self._transition('retreat',ctx)
            return self._output()
        if self.state=='settle_released_corridor':
            if self._quiet(ctx,.4,max_tilt=18.):
                if (getattr(ctx,'policy_command',None) is not None and
                        not observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)):
                    # The existing checked withdrawal can add the missing
                    # room. Do not certify or stay latched at its first free
                    # cell, and do not alter the box/unknown occupancy.
                    self.last_release_probe=ctx.now
                    self._transition('retreat',ctx,'stopped_release_pose_needs_braking_reserve')
                    return self._output()
                if self._recovered_passage_effect(ctx,along):return self._output()
                if self._begin_release_pose_recovery(ctx,along):return self._output()
                self.last_release_probe=ctx.now
                self._transition('retreat',ctx,'corridor_changed_during_release')
                return self._output()
            if ctx.now-self.state_since>self.config.max_push_braking_seconds:
                return self._retry(ctx,'released_corridor_not_settled')
            return self._output()
        if self.state=='recover_release_pose':
            self.release_recovery_diagnostic={'time_s':float(ctx.now),'position':np.asarray(ctx.position).tolist(),
                'heading_rad':float(ctx.heading),'previous_command':np.asarray(ctx.limiter.command).tolist(),
                'filtered_measured':np.asarray(ctx.limiter.measured).tolist()}
            if ctx.now-self.state_since>self.config.max_preparation_recovery_seconds:
                return self._fail(ctx,'release_pose_recovery_timeout')
            if self.release_axis_pending:
                self.release_recovery_diagnostic.update(reason='waiting_for_axis_recovery_stop',attempted_axes=list(self.release_axis_attempts))
                if not self._quiet(ctx,.4,max_tilt=18.):return self._output()
                if self.release_pose_recoveries>=self.config.max_retries:
                    self.release_axis_pending=False
                    self.release_recovery_diagnostic['reason']='axis_recovery_budget_exhausted'
                    return self._output()
                withdrawn=float(np.dot(self.release_origin-np.asarray(ctx.position[:2]),self.direction))
                choice=body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading,
                    excluded=self.release_axis_attempts,maximum_distance=max(0.,self.config.release_distance-withdrawn))
                self.release_axis_pending=False;self.progress_mark=np.asarray(ctx.position[:2]).copy();self.progress_since=ctx.now
                if choice is None:
                    self.release_recovery_diagnostic['reason']='no_safe_body_axis_alternative'
                    return self._output()
                tracker=ForwardExplorationPathTracker(ctx.position[:2],[tuple(self.task_goal)],guard_motion=False)
                if not tracker.set_checked_buffer_target(ctx.position,ctx.heading,ctx.planner,ctx.boxes,choice['target']):
                    self.release_recovery_diagnostic['reason']='axis_target_changed_before_handoff'
                    return self._output()
                self.release_axis_attempts.append(choice['axis']);self.release_pose_recoveries+=1
                self.release_recovery_tracker=tracker;self.release_recovery_target=choice['target'].copy()
                self.events.append({'skill':self.spec.name,'object_id':self.plan.box_id,'from':self.state,'to':self.state,
                    'reason':'measured_stall_select_body_axis','time_s':float(ctx.now),'retry':self.retries,
                    'axis':choice['axis'],'recovery_target':choice['target'].tolist(),'distance_m':choice['distance_m'],
                    'release_pose_recovery':self.release_pose_recoveries,'original_recovery_started_at':float(self.state_since)})
                self.release_recovery_diagnostic.update(reason='checked_body_axis_selected',axis=choice['axis'],target=choice['target'].tolist())
                return self._output()
            if self.progress_mark is None or self.progress_since is None:
                self.progress_mark=np.asarray(ctx.position[:2]).copy();self.progress_since=ctx.now
            if np.linalg.norm(np.asarray(ctx.position[:2])-self.progress_mark)>=.025:
                self.progress_mark=np.asarray(ctx.position[:2]).copy();self.progress_since=ctx.now
            translating=(getattr(self.release_recovery_tracker,'_buffer_active',False)
                and getattr(self.release_recovery_tracker,'_buffer_turn_target',None) is None)
            if (getattr(ctx,'policy_command',None) is not None and translating
                    and self.release_pose_recoveries<self.config.max_retries and ctx.now-self.progress_since>=1.5
                    and np.linalg.norm(ctx.body_velocity[:2])<.03 and np.linalg.norm(ctx.limiter.command[:2])>.20):
                self.release_axis_pending=True;self.settled_since=None
                self.release_recovery_diagnostic['reason']='stop_after_measured_translation_stall'
                self.events.append({'skill':self.spec.name,'object_id':self.plan.box_id,'from':self.state,'to':self.state,
                    'reason':'recovery_translation_stalled','time_s':float(ctx.now),'retry':self.retries,
                    'progress_m':float(np.linalg.norm(np.asarray(ctx.position[:2])-self.progress_mark)),
                    'stall_seconds':float(ctx.now-self.progress_since),'original_recovery_started_at':float(self.state_since)})
                return self._output()
            current_free=body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
            self.release_recovery_diagnostic['nominal_body_free']=bool(current_free)
            maneuver_active=(getattr(self.release_recovery_tracker,'_buffer_active',False)
                or getattr(self.release_recovery_tracker,'_buffer_turn_target',None) is not None)
            self.release_recovery_diagnostic['latched_maneuver_active']=bool(maneuver_active)
            if current_free and not maneuver_active:
                measured=np.asarray(ctx.limiter.measured,float)
                # A free present pose alone is insufficient while the base
                # is still moving toward a wall. Keep the original measured
                # stopping sweep, then settle and recheck the fresh map.
                braking_clear=body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,
                        measured[0],measured[2],horizon=.8,lateral=measured[1])
                self.release_recovery_diagnostic.update(reason='nominal_body_free',measured_braking_clear=bool(braking_clear))
                if braking_clear:
                    self.release_recovery_tracker=None;self.release_recovery_target=None
                    self._transition('settle_released_corridor',ctx,'release_body_space_recovered')
                return self._output()
            action,tracking=self.release_recovery_tracker.action(ctx.position[:2],ctx.heading,ctx.planner,ctx.boxes)
            self.release_recovery_diagnostic.update(tracker=dict(tracking),raw_action=dict(action))
            if tracking.get('needs_replan') or not tracking.get('recovering_buffer'):
                self.release_recovery_diagnostic['reason']='tracker_handoff_or_replan'
                return self._output()
            self.release_recovery_target=np.asarray(tracking['target'][:2])
            action=smooth_recovery_action(action,ctx.position[:2],ctx.heading,ctx.planner,
                ctx.boxes,ctx.limiter,ctx.now,directional_translation=True)
            self.release_recovery_diagnostic.update(reason='smoothed_recovery',smoothing=ctx.limiter.recovery_diagnostic)
            return self._output(action['move_by_speed'])
        if self.state=='settle_blocked_retreat':
            # A rejected withdrawal sample can arrive after several seconds
            # of real retreat. Brake before asking the existing overlap
            # recovery to move; its measured-stop prerequisite must get an
            # opportunity to become true. Keep all geometry/momentum checks.
            if self._quiet(ctx,.4,max_tilt=18.):
                if self._begin_release_pose_recovery(ctx,along,blocked_contact=True):
                    return self._output()
                options=({'speed':self.config.reverse_speed} if getattr(ctx,'policy_command',None) is not None else {})
                answer=observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,
                    self.direction,ctx.limiter,ctx.now,**options)
                if answer is not None:
                    self._transition('retreat',ctx,'retry_withdrawal_after_measured_stop')
                    return self._output()
                # Count an unavailable recovery from this actual blocked
                # episode, rather than from the start of successful motion.
                if ctx.now-self.state_since>5:
                    return self._fail(ctx,'no_safe_observed_retreat')
            elif ctx.now-self.state_since>self.config.max_push_braking_seconds:
                return self._fail(ctx,'blocked_retreat_not_settled')
            return self._output()
        if self.state=='retreat':
            if ctx.now-self.last_release_probe>=.25:
                self.last_release_probe=ctx.now
                # A barely clear moving pose can relax toward the box after
                # braking. Keep a checked reserve for the verified 45D gait.
                release_ready=(getattr(ctx,'policy_command',None) is None or
                    observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading))
                if release_ready and self._observed_passage_effect(ctx,along):
                    self._transition('settle_released_corridor',ctx,'observed_corridor_and_free_release_pose')
                    return self._output()
            withdrawn=float(np.dot(self.release_origin-np.asarray(ctx.position[:2]),self.direction))
            target_gap=self.config.front_gap if self.release_reason!='side_parked' else self.config.front_gap+.25
            if gap>=target_gap or withdrawn>=self.config.release_distance:
                self._transition('settle_after_retreat',ctx);return self._output()
            # The verified Unitree profile uses the same configured reverse
            # speed as preparation. Legacy callers retain the 0.20 default.
            retreat_options=({'speed':self.config.reverse_speed} if getattr(ctx,'policy_command',None) is not None else {})
            answer=observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,self.direction,ctx.limiter,ctx.now,**retreat_options)
            if answer is not None:
                self.settled_since=None
                return self._output(answer[0])
            if self._quiet(ctx,.4,max_tilt=18.) and self._begin_release_pose_recovery(ctx,along,blocked_contact=True):
                return self._output()
            self._transition('settle_blocked_retreat',ctx,'withdrawal_motion_check_blocked')
            # Preserve an already quiet first sample in the new settling
            # phase; moving samples cannot start this measured hold.
            self._quiet(ctx,.4,max_tilt=18.)
            return self._output()
        if self.state=='settle_after_retreat':
            if self._quiet(ctx,.4,max_tilt=18.):
                if self.release_reason!='side_parked' and self._recovered_passage_effect(ctx,along):
                    return self._output()
                if self.terminal_release_reason is not None:return self._fail(ctx,self.terminal_release_reason)
                if self.release_reason=='side_parked':
                    ctx.layers.refresh_global(reason='skill_release_observe');self.last_verify=-1e9
                    self._transition('verify_passage',ctx)
                else:
                    self.release_origin=None;self.turn=None;self.anchor=None
                    self._transition('prepare',ctx,'retry_from_clear_preparation_pose')
            return self._output()
        if self.state=='verify_passage':
            if ctx.now-self.last_verify>=.25:
                self.last_verify=ctx.now
                tracked=next((box for box in ctx.layers.current_boxes if box.id==self.plan.box_id),None)
                if tracked is not None:
                    proof,route,full=observed_passage_routes(ctx.layers,tracked,self.task_start,self.task_goal,
                        np.asarray(ctx.position[:2]),ctx.layers.current_boxes)
                    if proof and route and full:
                        self.certificate=proof;self.status='succeeded';self.reason='observed_passage_created'
                        self._transition('succeeded',ctx,self.reason);return self._output()
            if ctx.now-self.state_since>self.config.max_verify_seconds:
                if self.observation_moves>=2:return self._fail(ctx,'passage_not_confirmed_after_new_views')
                hint=getattr(ctx.layers,'passage_observation_hint',None)
                if hint and hint.get('object_id')!=self.plan.box_id:hint=None
                passage=hint or ctx.planner.passages.get(self.plan.box_id)
                options=[]
                if passage:
                    center=np.asarray(passage['center']);axis=np.asarray(passage['axis'])
                    options.extend(center-axis*distance for distance in [.8,1.,.6])
                options.extend([np.asarray(ctx.position[:2])-.30*self.direction,
                                np.asarray(ctx.position[:2])-.30*self.axis])
                target=None
                with ctx.layers.global_planning() as boxes:
                    saved_heading=ctx.planner.robot_heading;ctx.planner.robot_heading=None
                    try:
                        for candidate in options:
                            if np.linalg.norm(candidate-np.asarray(ctx.position[:2]))<.20:continue
                            if not body_rotation_clear(ctx.planner,boxes,candidate,ctx.heading,ctx.heading,margin=.04):continue
                            if ctx.planner.path(ctx.position[:2],candidate,boxes):target=candidate;break
                    finally:ctx.planner.robot_heading=saved_heading
                if target is None:return self._fail(ctx,'no_safe_reobservation_pose')
                self.navigation_goal=target;self.observation_moves+=1
                self._transition('reobserve',ctx,'change_view_after_box_motion')
                return self._output(navigation_goal=target)
            return self._output()
        if self.state=='reobserve':
            if not ctx.navigation_reached:return self._output(navigation_goal=self.navigation_goal)
            ctx.layers.refresh_global(reason='skill_reobservation_pose')
            self._transition('verify_passage',ctx);return self._output()
        return self._fail(ctx,'unknown_skill_state')

    def feedback(self,ctx):
        if ctx.box is None:return {}
        gap,lateral,yaw_error=self._box_frame(ctx)
        return {'nose_gap_m':gap,'nose_lateral_error_m':lateral,'heading_error_rad':yaw_error,
                'body_velocity_mps':list(map(float,ctx.body_velocity[:2])),
                'measured_yaw_rate_rad_s':float(ctx.yaw_rate),'tilt_degrees':float(ctx.tilt_degrees),
                'body_angular_velocity_rad_s':None if getattr(ctx,'body_angular_velocity',None) is None else list(map(float,ctx.body_angular_velocity)),
                'push_braking_cycles':self.push_brakes,'measured_box_speed_mps':float(self.box_velocity),
                'remaining_push_m':None if self.initial_box is None else float(self.plan.push_distance-np.dot(np.asarray(ctx.box.xy)-self.initial_box,self.direction)),
                'stage_xy':None if self.stage is None else self.stage.tolist(),
                'preparation_recovery_target':None if self.recovery_target is None else self.recovery_target.tolist(),
            'release_recovery_target':None if self.release_recovery_target is None else self.release_recovery_target.tolist(),
            'release_pose_recoveries':self.release_pose_recoveries,
            'preparation_attitude_hold':bool(self.state in PREPARATION_SETTLING_STATES and ctx.tilt_degrees>=18.),
                'heading_trim':None if self.heading_trim is None else self.heading_trim.last_detail,
                'anchor_xy':None if self.anchor is None else self.anchor.tolist()}

    def report(self):
        moved=0. if self.initial_box is None or self.last_box is None else float(np.dot(self.last_box-self.initial_box,self.direction))
        return {'name':self.spec.name,'version':self.spec.version,'status':self.status,'reason':self.reason,
            'object_id':self.plan.box_id,'recognized_category':self.permission.category,
            'interaction_source':self.permission.source,'source_reference':self.permission.source_reference,
            'source_digest':self.permission.source_digest,
            'recognition_frames':list(self.permission.evidence_frames),'classifier_models':list(self.permission.classifier_models),
            'push_braking_cycles':self.push_brakes,'retries':self.retries,'preparation_recoveries':self.preparation_recoveries,'actual_push_distance_m':moved,'maximum_tilt_degrees':self.max_tilt,
            'passage_certificate':self.certificate,'object_pose_feedback':'selected permitted simulator rigid body'}
