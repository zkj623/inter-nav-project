"""Short, body-checked pose sequences connecting a released box to its corridor."""
import math
import copy
from dataclasses import fields
import numpy as np

from grutopia_extension.interactive_navigation.go2_passage_guidance import _straight_body_connector_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear,navigation_footprint_samples
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate
from grutopia_extension.interactive_navigation.local_staging_control import StagingYawRate,TurnReversalGate,TurnPositionHold,native_translation_speed
from grutopia_extension.interactive_navigation.go2_turn_response import TurnResponseAssist
from grutopia_extension.interactive_navigation.go2_heading_trim import YawStepPulse


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


class PassageConnectionAdmission:
    """Wait for a genuinely quiet release instead of sampling readiness once."""
    def __init__(self,now):
        self.started=float(now);self.stop=MeasuredStopGate()

    def update(self,position,linear_velocity,angular_velocity,tilt,now):
        quiet=(np.linalg.norm(linear_velocity[:2])<.06 and abs(linear_velocity[2])<.05
               and np.max(np.abs(angular_velocity))<.10 and 0<=tilt<15.)
        timeout=now-self.started>6.
        if not quiet:self.stop.reset()
        ready=not timeout and quiet and self.stop.update(position,linear_velocity,angular_velocity,tilt,now)
        return {'ready':bool(ready),'timed_out':bool(timeout),'quiet':bool(quiet),'elapsed_seconds':float(now-self.started)}


def _rotation_clear(planner,boxes,position,first,last,margin):
    # Keep angular samples finer than the body-grid spacing, even for a
    # half-turn. The original 19-sample check is used on each short interval.
    angles=np.linspace(first,last,max(2,int(math.ceil(abs(last-first)/.35))+1))
    return all(body_rotation_clear(planner,boxes,position,a,b,margin=margin)
               for a,b in zip(angles,angles[1:]))


def _executable_translation_clear(planner,boxes,start,heading,target,*,reserve=0.,heading_tolerance=0.):
    start=np.asarray(start[:2],float);target=np.asarray(target[:2],float)
    delta=target-start;distance=float(np.linalg.norm(delta))
    if distance<.025:return _straight_body_connector_clear(planner,boxes,start,heading,target,margin=.04+reserve)
    direction=delta/distance
    for angle in sorted(set([heading-heading_tolerance,heading,heading+heading_tolerance])):
        c,s=math.cos(angle),math.sin(angle)
        body_direction=np.array([c*direction[0]+s*direction[1],-s*direction[0]+c*direction[1]])
        speed=native_translation_speed(.35,body_direction)
        length=max(distance+.025,.60*speed+speed*speed/(2*.65))
        if not _straight_body_connector_clear(planner,boxes,start,angle,target,margin=.04+reserve):return False
        if not _straight_body_connector_clear(planner,boxes,start,angle,start+length*direction,margin=.04):return False
    return True


def _reverse_diagonal_shift(start,heading,target):
    delta=np.asarray(target[:2],float)-np.asarray(start[:2],float)
    c,s=math.cos(heading),math.sin(heading)
    forward=c*delta[0]+s*delta[1];lateral=-s*delta[0]+c*delta[1]
    return (-forward>max(.04,.25*abs(lateral)) and abs(lateral)>max(.04,.25*abs(forward)))


def _translation_waypoints(planner,boxes,start,heading,target,*,reserve=0.,heading_tolerance=0.):
    """Replace poorly executed reverse/sideways combinations by checked axes.

    Keep the original straight-route admission and check BOTH new legs with
    the same full-body, unknown-space, stride and braking conditions. Prefer
    sidestepping first; if neither ordering fits, reject this candidate.
    """
    start=np.asarray(start[:2],float);target=np.asarray(target[:2],float)
    if not _executable_translation_clear(planner,boxes,start,heading,target,
            reserve=reserve,heading_tolerance=heading_tolerance):return None
    if not _reverse_diagonal_shift(start,heading,target):return (target,)
    c,s=math.cos(heading),math.sin(heading);forward=np.array([c,s]);side=np.array([-s,c])
    delta=target-start
    for middle in [start+side*np.dot(delta,side),start+forward*np.dot(delta,forward)]:
        if all(_executable_translation_clear(planner,boxes,a,heading,b,reserve=reserve,
                heading_tolerance=heading_tolerance) for a,b in [(start,middle),(middle,target)]):
            return (middle,target)
    return None


def _lane_uses_vacated_space(planner,certificate,center,axis):
    if 'original_box' not in certificate:return False,None
    old_axis=np.asarray(certificate['axis'],float);old_axis/=np.linalg.norm(old_axis)
    normal=np.array([-axis[1],axis[0]])
    rear=min(float(np.dot(x*axis+y*normal,old_axis)) for x in [-.32,.50] for y in [-.25,.25])
    denominator=float(np.dot(axis,old_axis))
    if denominator<.80:return False,None
    boundary=float(np.dot(certificate['center'],old_axis))
    cross=(boundary+.02-rear-float(np.dot(center,old_axis)))/denominator
    names={f.name for f in fields(BoxState)}
    original=BoxState(**{k:v for k,v in certificate['original_box'].items() if k in names})
    mask=planner.box_mask(original)
    if mask is None:return False,None
    xx,yy=navigation_footprint_samples(planner,margin=.04)
    row,col=planner.cell(center+cross*axis)
    columns=col+np.rint((axis[0]*xx-axis[1]*yy)/planner.resolution).astype(int)
    rows=row+np.rint((axis[1]*xx+axis[0]*yy)/planner.resolution).astype(int)
    if rows.min()<0 or columns.min()<0 or rows.max()>=mask.shape[0] or columns.max()>=mask.shape[1]:return False,None
    return bool(mask[rows,columns].any()),cross


def _plan_pose_lane(position,heading,certificate,planner,boxes,axis,center,exit_point,cross,reserve,rotation_cache):
    """Search a bounded turn/shift/align connection, never modify map evidence.

    The first shift makes room for the remaining rotation. The second shift
    joins the certified centerline while holding its heading. Each complete
    rotation and each crossed base-cell interval has the original 4 cm margin.
    """
    if (not certificate.get('side_parked') or not certificate.get('original_box_would_block_corridor')
            or certificate.get('confirming_observation_time_s',-math.inf)<=certificate.get('last_box_motion_time_s',math.inf)):
        return None
    start=np.asarray(position[:2],float);axis=np.asarray(axis,float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:return None
    axis=axis/np.linalg.norm(axis);center=np.asarray(center,float);exit_point=np.asarray(exit_point,float)
    normal=np.array([-axis[1],axis[0]]);course=math.atan2(axis[1],axis[0])
    error=wrap(course-heading);choices=[]
    def rotation_clear(point,first,last,margin):
        key=(*planner.cell(point),float(first),float(last),float(margin))
        if key not in rotation_cache:rotation_cache[key]=_rotation_clear(planner,boxes,point,first,last,margin)
        return rotation_cache[key]
    limit=min(.5,cross-.03)
    # Every candidate uses this suffix. Reject blocked lanes before the
    # much larger approach-pose search; this does not omit any body sample.
    suffix=center+limit*axis
    if not all(_straight_body_connector_clear(planner,boxes,suffix,course+e,exit_point,margin=.04+reserve)
               for e in [-.08,0.,.08]):return None
    aligned_options=[]
    for longitudinal in np.arange(-.65,min(.51,cross-.03),.10):
        aligned=center+longitudinal*axis
        if _executable_translation_clear(planner,boxes,aligned,course,exit_point,reserve=reserve,heading_tolerance=.08):
            aligned_options.append(aligned)
    if not aligned_options:return None
    for first_heading in np.linspace(heading,heading+error,13):
        first_turn=abs(first_heading-heading)
        if .01<first_turn<.15:continue  # Avoid a path that depends on an unretained micro-turn.
        first_end=first_heading if first_turn<.01 else first_heading+math.copysign(.08,first_heading-heading)
        if not rotation_clear(start,heading,first_end,.04):continue
        for aligned in aligned_options:
            for lateral in [-.375,-.30,-.225,-.15,-.075,0.,.075,.15,.225,.30,.375]:
                middle=aligned+lateral*normal
                first_distance=float(np.linalg.norm(middle-start))
                if not .04<first_distance<=.95:continue
                last_heading=first_heading+wrap(course-first_heading)
                first_path=_translation_waypoints(planner,boxes,start,first_heading,middle,
                    heading_tolerance=.08 if first_turn>.01 else 0.)
                if first_path is None:continue
                if not rotation_clear(middle,min(first_heading,last_heading)-.08,
                    max(first_heading,last_heading)+.08,.04+reserve):continue
                if not _executable_translation_clear(planner,boxes,middle,last_heading,aligned,reserve=reserve,heading_tolerance=.08):continue
                first_length=sum(float(np.linalg.norm(b-a)) for a,b in zip((start,*first_path[:-1]),first_path))
                distance=first_length+float(np.linalg.norm(aligned-middle)+np.linalg.norm(exit_point-aligned))
                cost=distance+.15*(abs(first_heading-heading)+abs(last_heading-first_heading))
                choices.append((cost,distance,first_heading,middle,last_heading,aligned,first_path))
    if not choices:return None
    cost,distance,first_heading,middle,last_heading,aligned,first_path=min(choices,key=lambda row:row[0])
    operations=[]
    if abs(wrap(first_heading-heading))>.035:operations.append({'kind':'turn','heading':float(first_heading)})
    for point in first_path:
        operations.append({'kind':'move','heading':float(first_heading),'target':point.tolist()})
    if abs(wrap(last_heading-first_heading))>.035:operations.append({'kind':'turn','heading':float(last_heading)})
    if np.linalg.norm(aligned-middle)>.025:operations.append({'kind':'move','heading':float(last_heading),'target':aligned.tolist()})
    operations.append({'kind':'move','heading':float(last_heading),'target':exit_point.tolist()})
    compact=[];last_position=start.copy()
    for operation in operations:
        if operation['kind']=='move' and compact and compact[-1]['kind']=='move':
            previous=compact[-1];first=np.asarray(previous['target'])-last_position
            second=np.asarray(operation['target'])-np.asarray(previous['target'])
            collinear=(abs(first[0]*second[1]-first[1]*second[0])<1e-7 and np.dot(first,second)>0)
            if (abs(wrap(previous['heading']-operation['heading']))<.01 and collinear
                    and _executable_translation_clear(planner,boxes,last_position,operation['heading'],operation['target'],heading_tolerance=.08)):
                compact[-1]=operation;continue
        if compact and compact[-1]['kind']=='move':last_position=np.asarray(compact[-1]['target'])
        compact.append(operation)
    operations=compact
    return {'operations':operations,'length_m':distance,'cost':cost,'body_margin_m':.04,
            'candidate_count':len(choices),'start':start.tolist(),'exit':exit_point.tolist(),
            'axis':axis.tolist(),'center':center.tolist(),'tracking_reserve_m':reserve,
            'heading_tolerance_rad':.08,'full_stride_and_braking_checked':True}


def plan_pose_connection(position,heading,certificate,planner,boxes):
    if not certificate.get('side_parked') or not certificate.get('original_box_would_block_corridor'):return None
    # Geometry is unchanged during this synchronous search. Fuse all admitted
    # boxes into a private map once instead of rebuilding their masks for
    # every candidate. No unknown cell or obstacle is removed.
    view=copy.copy(planner);view.free=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return None
        view.free &= ~mask
    planner=view;boxes=()
    old_axis=np.asarray(certificate['axis'],float).copy();old_axis/=np.linalg.norm(old_axis)
    old_center=np.asarray(certificate['center'],float)
    old_exit=np.asarray(certificate.get('guide_exit',certificate['corridor_end']),float)
    exit_distance=float(np.dot(old_exit-old_center,old_axis))
    if exit_distance<.4:return None
    original=math.atan2(old_axis[1],old_axis[0])
    goal_delta=np.asarray(certificate['route_goal'],float)-old_center
    goal_angle=math.atan2(goal_delta[1],goal_delta[0])
    angles=[original,original-.174533,original+.174533,original-.349066,original+.349066]
    if abs(wrap(goal_angle-original))<.52:angles.append(goal_angle)
    cardinal=round(original/(math.pi/2))*(math.pi/2)
    if abs(wrap(cardinal-original))<.52:angles.append(cardinal)
    angles=sorted(set(round(a,6) for a in angles))
    rotation_cache={}
    for reserve in [.04,0.]:
        choices=[]
        for angle in angles:
            axis=np.array([math.cos(angle),math.sin(angle)]);normal=np.array([-axis[1],axis[0]])
            for lateral in np.arange(-.15,.151,.025):
                center=old_center+lateral*normal;exit_point=center+exit_distance*axis
                uses,cross=_lane_uses_vacated_space(planner,certificate,center,axis)
                if not uses or not .05<cross<exit_distance-.02:continue
                result=_plan_pose_lane(position,heading,certificate,planner,boxes,axis,center,exit_point,cross,reserve,rotation_cache)
                if result is not None:
                    result['lane_offset_m']=float(lateral);choices.append(result)
        if choices:
            result=min(choices,key=lambda p:p['cost'])
            result['reference_passage']={'axis':old_axis.tolist(),'center':old_center.tolist()}
            return result
    return None


class PoseConnectionControl:
    """Execute a checked pose sequence with actual progress and measured stops.

    Effective translation targets are applied before slew and body checks.
    Small transient commands during acceleration/braking remain legal; the
    controller never raises an already checked velocity afterwards.
    """
    def __init__(self,plan,now,*,started=None):
        self.plan=copy.deepcopy(plan);self.index=0;self.started=float(now if started is None else started)
        self.phase='begin';self.stop=MeasuredStopGate();self.braking=False
        self.pending_failure=None;self.failed=None;self.done=False
        self.rate=StagingYawRate();self.reversal=TurnReversalGate();self.assist=TurnResponseAssist()
        self.progress_position=None;self.progress_time=float(now);self.operation_started=float(now)
        self.stop_attempts=0
        self.move_origin=None;self.move_direction=None;self.minimum_progress=0.
        self.turn_position_hold=TurnPositionHold()
        self.yaw_step=None
        self.axis_splits=0

    def _brake(self,reason=None):
        if not self.braking:
            self.stop.reset()
            # An intentional brake/settle is not a failed yaw request.
            # Start a new response window when a fine correction resumes.
            self.assist=TurnResponseAssist()
        if reason is not None:self.turn_position_hold.reset()
        self.yaw_step=None
        self.braking=True
        if reason is not None and self.pending_failure is None:self.pending_failure=reason

    def _start_move(self,position,target):
        delta=np.asarray(target)-np.asarray(position[:2]);distance=float(np.linalg.norm(delta))
        self.move_origin=np.asarray(position[:2],float).copy()
        self.move_direction=delta/max(distance,1e-9)
        self.minimum_progress=min(.04,max(0.,distance-.03))

    def _next_motion_ready(self,position,heading,planner,boxes):
        """Accept an intermediate region only by checking its actual continuation."""
        remaining=self.plan['operations'][self.index+1:]
        if not remaining:
            reference=self.plan.get('reference_passage')
            if reference is None:return False
            axis=np.asarray(reference['axis']);axis/=np.linalg.norm(axis)
            forward=np.array([math.cos(heading),math.sin(heading)]);side=np.array([-forward[1],forward[0]])
            rear=min(float(np.dot(x*forward+y*side,axis)) for x in [-.32,.50] for y in [-.25,.25])
            return (np.dot(np.asarray(position[:2])-reference['center'],axis)+rear>.02
                and _straight_body_connector_clear(planner,boxes,position,heading,position,margin=.04))
        current_heading=float(heading)
        for operation in remaining:
            target_heading=current_heading+wrap(operation['heading']-current_heading)
            if operation['kind']=='turn':
                error=target_heading-current_heading
                if abs(error)>=.08:
                    end=target_heading+math.copysign(.08,error)
                    if not _rotation_clear(planner,boxes,position,current_heading,end,.04):return False
                    current_heading=target_heading
            else:
                if abs(wrap(target_heading-current_heading))>.15:return False
                low=min(current_heading,target_heading-.08);high=max(current_heading,target_heading+.08)
                return _executable_translation_clear(planner,boxes,position,(low+high)/2,
                    operation['target'],heading_tolerance=(high-low)/2)
        return False

    def update(self,position,heading,planner,boxes,limiter,now,linear_velocity,angular_velocity,tilt):
        position=np.asarray(position,float)
        state={'controller':'checked_passage_pose_connection','operation_index':self.index,
               'operation_count':len(self.plan['operations']),'elapsed_seconds':float(now-self.started)}
        def checked(request,target=None):
            command,safe=checked_contact_turn_command(request,position,heading,planner,boxes,limiter,now,target_position=target)
            state.update(desired_command=np.asarray(request).tolist(),applied_command=command.tolist(),body_motion_safe=bool(safe))
            if not safe:self._brake('pose_connection_motion_blocked')
            return command
        if self.done or self.failed:return np.zeros(3),state
        if now-self.started>45.:self._brake('pose_connection_deadline')
        if tilt>=15.:self._brake('pose_connection_attitude')
        operation=self.plan['operations'][self.index]
        target=None if operation['kind']=='turn' else np.asarray(operation['target'],float)
        error=wrap(operation['heading']-heading)
        if (self.phase=='begin' and not self.braking and target is not None and abs(error)<=.12
                and _reverse_diagonal_shift(position,operation['heading'],target)):
            waypoints=_translation_waypoints(planner,boxes,position,operation['heading'],target,
                heading_tolerance=max(.08,abs(error)))
            if waypoints is None:self._brake('pose_connection_axis_sequence_blocked')
            elif self.axis_splits>=4:self._brake('pose_connection_axis_split_limit')
            else:
                self.axis_splits+=1
                replacement=[{'kind':'move','heading':operation['heading'],'target':point.tolist()} for point in waypoints]
                self.plan['operations'][self.index:self.index+1]=replacement
                operation=self.plan['operations'][self.index];target=np.asarray(operation['target'],float)
                state['axis_split_from_actual_pose']={'start':position[:2].tolist(),'operations':replacement}
        state.update(operation_count=len(self.plan['operations']),axis_splits=self.axis_splits)
        distance=0. if target is None else float(np.linalg.norm(target-position[:2]))
        state.update(kind=operation['kind'],target=None if target is None else target.tolist(),heading_error_rad=error,distance_m=distance)
        if self.phase=='begin' and not self.braking:
            self.operation_started=float(now);self.progress_position=position[:2].copy();self.progress_time=float(now)
            self.stop_attempts=0;self.rate.reset();self.reversal=TurnReversalGate();self.assist=TurnResponseAssist()
            self.turn_position_hold.reset()
            if target is None:
                if not body_rotation_clear(planner,boxes,position,heading,heading+error,margin=.04):
                    self._brake('pose_connection_turn_changed')
                elif .08<=abs(error)<.25:
                    self.yaw_step=YawStepPulse(error,heading,now)
            elif abs(error)>.12 or not _straight_body_connector_clear(planner,boxes,position,heading,target):
                self._brake('pose_connection_segment_changed')
            if target is not None:self._start_move(position,target)
            self.phase='execute'
        if self.braking:
            state['phase']='measured_stop'
            command=checked([0.,0.,0.])
            quiet_command=np.linalg.norm(command[:2])<.03 and abs(command[2])<.06
            if not quiet_command:self.stop.reset()
            elif self.stop.update(position,linear_velocity,angular_velocity,tilt,now):
                exact=(target is None and abs(error)<.08) or (target is not None and distance<.04)
                region=(target is not None and distance<.15 and self._next_motion_ready(position,heading,planner,boxes))
                state['accepted_execution_region']=bool(region)
                if self.pending_failure and not (self.pending_failure=='pose_connection_motion_blocked' and region):
                    self.failed=self.pending_failure
                elif exact or region:
                    self.pending_failure=None
                    self.index+=1;self.braking=False;self.phase='begin';self.stop.reset()
                    if self.index==len(self.plan['operations']):self.done=True
                else:
                    self.stop_attempts+=1
                    if self.stop_attempts>=4:self.failed='pose_connection_settled_short'
                    else:
                        self.braking=False;self.stop.reset();self.rate.reset();self.reversal=TurnReversalGate()
                        self.progress_time=float(now);self.progress_position=position[:2].copy()
                        if target is not None:self._start_move(position,target)
                        elif abs(error)<.25:self.yaw_step=YawStepPulse(error,heading,now)
            state.update(done=self.done,failure=self.failed)
            return command,state
        if now-self.operation_started>20.:
            self._brake('pose_connection_operation_deadline');return checked([0.,0.,0.]),state
        if target is None:
            if self.yaw_step is not None:
                request,detail=self.yaw_step.update(heading,limiter.policy_command,now)
                state.update(phase='acknowledged_heading_step',yaw_step=detail)
                if self.yaw_step.complete or self.yaw_step.failed:
                    failure=self.yaw_step.failed
                    self._brake(failure)
                else:
                    request=self.turn_position_hold.apply(request,position,heading,limiter.measured)
                    state['turn_position_hold_command']=request[:2].tolist()
                # Do not mistake a transient body twist for a retained turn
                # before the established .55/.04s input reaches the actor.
                return checked(request),state
            measured=float(limiter.measured[2])
            stopping_angle=abs(measured)*.25+limiter.command[2]**2/(2*limiter.yaw_acceleration)+.03
            reverse=self.reversal.update(error,measured,heading,now)
            if abs(error)<.08 or (error*measured>0 and abs(measured)>.1 and abs(error)<stopping_angle) or reverse:
                self._brake();return checked([0.,0.,0.]),state
            request=np.array([0.,0.,self.rate.request(error,measured,now,native=True)])
            request,assist=self.assist.update(request,position,heading,planner,boxes,limiter,now,maintain_arc=True)
            request=self.turn_position_hold.apply(request,position,heading,limiter.measured)
            state['turn_position_hold_command']=request[:2].tolist()
            state.update(assist,phase='continuous_turn')
            if assist.get('turn_assist')=='no_observed_safe_arc':
                self._brake('pose_connection_gait_turn_blocked');return checked([0.,0.,0.]),state
            return checked(request),state
        if abs(error)>.15:
            self._brake('pose_connection_heading_drift');return checked([0.,0.,0.]),state
        delta=target-position[:2];direction=delta/max(distance,1e-9)
        c,s=math.cos(heading),math.sin(heading)
        rotation=np.array([[c,-s],[s,c]])
        toward=float(np.dot(rotation@np.asarray(linear_velocity[:2]),direction))
        commanded=max(0.,float(np.dot(rotation@limiter.command[:2],direction)))
        stopping_distance=.025+.5*max(0.,toward)+commanded**2/(2*limiter.linear_acceleration)
        progress=float(np.dot(position[:2]-self.move_origin,self.move_direction))
        state.update(progress_since_stop_m=progress,minimum_progress_before_goal_brake_m=self.minimum_progress)
        # A stationary gait must first receive its effective input. A command-
        # only braking radius otherwise stops short moves before any step.
        if distance<.03 or (toward>.04 and distance<stopping_distance and progress>=self.minimum_progress):
            self._brake();return checked([0.,0.,0.]),state
        if float(np.dot(position[:2]-self.progress_position,self.move_direction))>.025:
            self.progress_position=position[:2].copy();self.progress_time=float(now)
        if now-self.progress_time>3.:
            self._brake('pose_connection_translation_stalled');return checked([0.,0.,0.]),state
        body_direction=rotation.T@direction
        speed=native_translation_speed(.35,body_direction)
        request=np.array([*(speed*body_direction),float(np.clip(2.*error-.35*angular_velocity[2],-.25,.25))])
        state.update(phase='held_heading_translation',effective_target_speed_mps=speed)
        return checked(request,target),state
