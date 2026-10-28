"""Parameterized Go2 side-push skill with staged alignment and map confirmation.

The skill emits velocity/navigation commands. It never sets robot or object
poses; object movement must come from the simulator's contact dynamics.
"""
from dataclasses import dataclass
import math
import numpy as np
from grutopia_extension.interactive_navigation.interaction_skill import SkillSpec,SkillCommand
from grutopia_extension.interactive_navigation.go2_contact_control import (
    ContactYawServo,contact_heading_hold,contact_pose_ready,observed_contact_retreat,
    push_centerline_velocity,push_completion_state)
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl,checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_rotation_clear,body_translation_recovery_clear,body_motion_clear)
from grutopia_extension.interactive_navigation.go2_passage_confirmation import observed_passage_routes


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class PushAsideConfig:
    front_gap:float=.35
    front_contact_reach:float=.17  # 0.50 m front-foot envelope minus 0.33 m nose offset
    lateral_speed:float=.35
    forward_speed:float=.25
    reverse_speed:float=.35
    push_start_speed:float=.30
    push_max_speed:float=.95
    push_acceleration:float=.15
    yaw_tolerance:float=.10
    lateral_tolerance:float=.055
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
        self.initial_box=None;self.last_box=None;self.stage=None;self.anchor=None;self.axis=None
        self.turn=None;self.yaw_servo=ContactYawServo(self.config.yaw_tolerance)
        self.settled_since=None;self.navigation_goal=None;self.events=[]
        self.release_origin=None;self.release_reason=None;self.certificate=None
        self.last_verify=-1e9;self.observation_moves=0;self.used_stages=[]
        self.progress_mark=None;self.progress_since=None;self.max_tilt=0.
        self.reason='';self.status='running';self.box_velocity=0.;self.last_box_time=None

    def _transition(self,state,ctx,reason=''):
        self.events.append({'skill':self.spec.name,'object_id':self.plan.box_id,'from':self.state,
            'to':state,'reason':reason,'time_s':float(ctx.now),'retry':self.retries})
        self.state=state;self.state_since=ctx.now;self.settled_since=None
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
        quiet=(np.linalg.norm(ctx.body_velocity[:2])<.08 and abs(ctx.yaw_rate)<.10 and ctx.tilt_degrees<max_tilt)
        if not quiet:self.settled_since=None;return False
        if self.settled_since is None:self.settled_since=ctx.now
        return ctx.now-self.settled_since>=seconds

    def _box_frame(self,ctx):
        extent=float(np.dot(np.abs(self.direction),np.asarray(ctx.box.dimensions[:2]))/2)
        nose=np.asarray(ctx.position[:2])+.33*np.array([math.cos(ctx.heading),math.sin(ctx.heading)])
        delta=np.asarray(ctx.box.xy)-nose
        return float(np.dot(delta,self.direction)-extent),float(np.dot(delta,self.tangent)),wrap(self.push_heading-ctx.heading)

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
                    points=[position]
                    points.extend(anchor-self.axis*back+self.direction*along
                                  for back in [.5,.7,.9,1.1] for along in [0.,.2,-.2,.4,-.4])
                    for point in points:
                        if any(np.linalg.norm(point-old)<.10 for old in self.used_stages):continue
                        if np.dot(np.asarray(ctx.box.xy)-point,self.direction)<extent+.5:continue
                        path=p.path(position,point,boxes) if np.linalg.norm(point-position)>.15 else [tuple(point)]
                        if not path:continue
                        arrival=ctx.heading if np.linalg.norm(point-position)<.15 else math.atan2(point[1]-path[-2][1],point[0]-path[-2][0]) if len(path)>1 else math.atan2(point[1]-position[1],point[0]-position[0])
                        if not body_rotation_clear(p,boxes,point,arrival,self.push_heading,margin=.05):continue
                        longitudinal=anchor+self.tangent*float(np.dot(point-anchor,self.tangent))
                        if not body_translation_recovery_clear(p,boxes,point,self.push_heading,longitudinal):continue
                        if not body_translation_recovery_clear(p,boxes,longitudinal,self.push_heading,anchor):continue
                        score=p.length(position,path)+.3*np.linalg.norm(point-anchor)
                        candidates.append((score,point.copy(),anchor.copy()))
                if not candidates:return False
                _,self.stage,self.anchor=min(candidates,key=lambda value:value[0])
                return True
            finally:p.robot_heading=heading

    def _retry(self,ctx,reason):
        self.retries+=1;self.release_reason=reason
        self.release_origin=None
        if self.retries>self.config.max_retries:return self._fail(ctx,'retry_limit:'+reason)
        self._transition('brake_for_recovery',ctx,reason)
        return self._output()

    def _translation(self,ctx,target,axis,speed,tolerance):
        delta=np.asarray(target)-np.asarray(ctx.position[:2]);error=float(np.dot(delta,axis))
        if abs(error)<tolerance:
            return np.zeros(3),self._quiet(ctx)
        self.settled_since=None
        c,s=math.cos(ctx.heading),math.sin(ctx.heading);rot=np.array([[c,-s],[s,c]])
        world_v=rot@np.asarray(ctx.body_velocity[:2]);unit=axis*math.copysign(1.,error)
        toward=float(np.dot(world_v,unit));lateral=abs(np.dot(axis,self.tangent))>.5
        if toward>.06 and abs(error)<.025+toward*(.65 if lateral else .4):return np.zeros(3),False
        world=unit*speed
        yaw_request=contact_heading_hold(wrap(self.push_heading-ctx.heading),ctx.yaw_rate)
        if not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.):yaw_request=0.
        request=[*(rot.T@world),yaw_request]
        command,_=checked_contact_turn_command(request,ctx.position,ctx.heading,ctx.planner,ctx.boxes,
            ctx.limiter,ctx.now,target_position=np.asarray(ctx.position[:2])+axis*error)
        return command,False

    def step(self,ctx):
        if self.status!='running':return self._output()
        if ctx.box is None or ctx.box.id!=self.plan.box_id:return self._fail(ctx,'target_track_lost')
        if self.started is None:
            self.started=ctx.now;self.state_since=ctx.now;self.initial_box=np.asarray(ctx.box.xy).copy()
        self.max_tilt=max(self.max_tilt,float(ctx.tilt_degrees))
        if not ctx.healthy or ctx.tilt_degrees>=30:return self._fail(ctx,'robot_unstable')
        if self.state in ['settle_at_stage','turn_at_stage','align_longitudinal','align_lateral','settle_contact'] and ctx.tilt_degrees>=18.:
            self.used_stages.append(np.asarray(ctx.position[:2]).copy())
            return self._retry(ctx,'preparation_attitude')
        if ctx.now-self.started>self.config.max_skill_seconds:return self._fail(ctx,'skill_timeout')
        moved=np.asarray(ctx.box.xy)-self.initial_box;along=float(np.dot(moved,self.direction))
        cross=float(np.dot(moved,self.tangent))
        if self.last_box is not None and ctx.now>self.last_box_time:
            observed=float(np.dot(np.asarray(ctx.box.xy)-self.last_box,self.direction))/(ctx.now-self.last_box_time)
            self.box_velocity=.8*self.box_velocity+.2*observed
        self.last_box=np.asarray(ctx.box.xy).copy();self.last_box_time=ctx.now
        if abs(cross)>.15:return self._fail(ctx,'box_left_safe_push_line')
        gap,lateral,yaw_error=self._box_frame(ctx)

        if self.state=='prepare':
            if not self._select_stage(ctx):return self._fail(ctx,'no_observed_preparation_pose')
            self.navigation_goal=self.stage;self._transition('navigate_to_stage',ctx)
            return self._output(navigation_goal=self.stage)
        if self.state=='navigate_to_stage':
            if not ctx.navigation_reached:return self._output(navigation_goal=self.stage)
            self._transition('settle_at_stage',ctx);return self._output()
        if self.state=='settle_at_stage':
            if self._quiet(ctx):
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
                self.yaw_servo=ContactYawServo(self.config.yaw_tolerance)
                self._transition('align_longitudinal',ctx);return self._output()
            return self._output(result[0])
        if self.state in ['align_longitudinal','align_lateral','settle_contact']:
            if abs(yaw_error)>.18:return self._retry(ctx,'heading_drift_during_alignment')
            if ctx.now-self.state_since>self.config.max_alignment_seconds:return self._retry(ctx,'alignment_timeout')
            actual_ready=contact_pose_ready(ctx.position,ctx.heading,ctx.box.xy,ctx.box.dimensions,self.direction,
                self.config.front_gap+.05,self.config.yaw_tolerance)
            if (self.state!='settle_contact' and actual_ready and body_motion_clear(
                    ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)):
                self._transition('settle_contact',ctx,'actual_contact_pose_ready')
                return self._output()
            if self.state=='align_longitudinal':
                target=self.anchor+self.tangent*float(np.dot(np.asarray(ctx.position[:2])-self.anchor,self.tangent))
                error=float(np.dot(target-np.asarray(ctx.position[:2]),self.direction))
                command,done=self._translation(ctx,target,self.direction,
                    self.config.forward_speed if error>=0 else self.config.reverse_speed,.05)
                if done:self._transition('align_lateral',ctx)
                return self._output(command)
            if self.state=='align_lateral':
                command,done=self._translation(ctx,self.anchor,self.tangent,self.config.lateral_speed,.035)
                if done:self._transition('settle_contact',ctx)
                return self._output(command)
            ready=contact_pose_ready(ctx.position,ctx.heading,ctx.box.xy,ctx.box.dimensions,self.direction,
                self.config.front_gap+.05,self.config.yaw_tolerance)
            if ready and self._quiet(ctx,.4):
                self.engage_box_position=np.asarray(ctx.box.xy).copy()
                self._transition('engage',ctx);return self._output()
            if abs(yaw_error)>self.config.yaw_tolerance:
                rate=self.yaw_servo.update(yaw_error,ctx.yaw_rate,ctx.now)
                command,_=checked_contact_turn_command([0,0,rate],ctx.position,ctx.heading,ctx.planner,ctx.boxes,ctx.limiter,ctx.now)
                return self._output(command)
            if abs(lateral)>self.config.lateral_tolerance:self._transition('align_lateral',ctx)
            elif gap>.475 or gap<.20:self._transition('align_longitudinal',ctx)
            return self._output()
        if self.state in ['engage','push']:
            if ctx.tilt_degrees>18:return self._retry(ctx,'contact_attitude')
            if abs(yaw_error)>.18 or abs(lateral)>.075:return self._retry(ctx,'contact_alignment_drift')
            if self.state=='engage' and (gap<=self.config.front_contact_reach+.02 or float(np.dot(np.asarray(ctx.box.xy)-self.engage_box_position,self.direction))>.015):
                self._transition('push',ctx);self.push_progress=along;self.push_progress_time=ctx.now
            clearance=ctx.planner.clearance(ctx.box)
            done=push_completion_state(None,clearance,along,True)
            if done and along>=self.plan.push_distance-.035:
                self.release_origin=np.asarray(ctx.position[:2]).copy();self.release_reason='side_parked'
                self._transition('brake_after_push',ctx,done);return self._output()
            if along>self.plan.push_distance+.25:return self._fail(ctx,'safe_push_distance_exceeded')
            elapsed=ctx.now-self.state_since
            if self.state=='engage' and elapsed>12:return self._retry(ctx,'contact_not_established')
            if self.state=='push':
                if along-self.push_progress>.015:self.push_progress=along;self.push_progress_time=ctx.now
                if ctx.now-self.push_progress_time>6:return self._retry(ctx,'box_motion_stalled')
            speed=self.config.push_start_speed if self.state=='engage' else min(self.config.push_max_speed,self.config.push_start_speed+self.config.push_acceleration*elapsed)
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
        if self.state=='retreat':
            withdrawn=float(np.dot(self.release_origin-np.asarray(ctx.position[:2]),self.direction))
            target_gap=self.config.front_gap if self.release_reason!='side_parked' else self.config.front_gap+.25
            if gap>=target_gap or withdrawn>=self.config.release_distance:
                self._transition('settle_after_retreat',ctx);return self._output()
            answer=observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,self.direction,ctx.limiter,ctx.now)
            if answer is not None:return self._output(answer[0])
            if ctx.now-self.state_since>5:return self._fail(ctx,'no_safe_observed_retreat')
            return self._output()
        if self.state=='settle_after_retreat':
            if self._quiet(ctx,.4,max_tilt=18.):
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

    def report(self):
        moved=0. if self.initial_box is None or self.last_box is None else float(np.dot(self.last_box-self.initial_box,self.direction))
        return {'name':self.spec.name,'version':self.spec.version,'status':self.status,'reason':self.reason,
            'object_id':self.plan.box_id,'recognized_category':self.permission.category,
            'interaction_source':self.permission.source,'source_reference':self.permission.source_reference,
            'source_digest':self.permission.source_digest,
            'recognition_frames':list(self.permission.evidence_frames),'classifier_models':list(self.permission.classifier_models),
            'retries':self.retries,'actual_push_distance_m':moved,'maximum_tilt_degrees':self.max_tilt,
            'passage_certificate':self.certificate,'object_pose_feedback':'selected permitted simulator rigid body'}
