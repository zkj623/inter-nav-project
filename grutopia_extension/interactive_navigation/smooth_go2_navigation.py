"""Continuous exploration velocity tracking with acceleration-bounded requests.

The original Go2 policy/actuator loop is unchanged. Local checks evaluate the
command actually sent after smoothing, including the full body footprint.
"""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear,body_translation_recovery_clear,navigation_body_blocked,navigation_footprint_samples,route_blocked,body_rotation_clear
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from scipy.ndimage import distance_transform_edt


class VelocitySlewLimiter:
    def __init__(self,linear_acceleration=.65,yaw_acceleration=1.8):
        self.linear_acceleration=linear_acceleration;self.yaw_acceleration=yaw_acceleration
        self.command=np.zeros(3);self.last_time=None
        self.measured=np.zeros(3);self.measurement_time=None
        self.staging=None;self.progress_position=None;self.progress_time=None
    def observe(self,velocity,now):
        velocity=np.asarray(velocity,float)
        if self.measurement_time is None:self.measured=velocity.copy()
        else:
            dt=max(0.,float(now)-self.measurement_time);alpha=dt/(.15+dt)
            self.measured+=alpha*(velocity-self.measured)
        self.measurement_time=float(now)
    def preview(self,target,now):
        dt=.005 if self.last_time is None else max(.001,min(.1,float(now)-self.last_time))
        target=np.asarray(target,float);delta=target[:2]-self.command[:2]
        maximum=self.linear_acceleration*dt
        planar=self.command[:2]+delta*min(1.,maximum/max(np.linalg.norm(delta),1e-9))
        yaw=self.command[2]+np.clip(target[2]-self.command[2],-self.yaw_acceleration*dt,self.yaw_acceleration*dt)
        return np.array([*planar,yaw])
    def accept(self,command,now):
        self.command=np.asarray(command,float).copy();self.last_time=float(now)


class SmoothExplorationTracker(ExplorationPathTracker):
    def __init__(self,start,path,limiter,*,max_forward=.50,max_lateral=.18):
        super().__init__(start,path,lookahead=.55,max_forward_speed=max_forward,max_lateral_speed=max_lateral)
        self.limiter=limiter;self.max_forward=max_forward;self.max_lateral=max_lateral
        self._last_plan_time=-1e9;self._desired=np.zeros(3)
        self._blocked_ticks=0;self._last_safe=np.zeros(3)
        self._observation_started=None
        self._observation_yaw_origin=0.;self._observation_yaw_sign=1.;self._observation_yaw_moving=False

    @property
    def staging_active(self):return self.limiter.staging is not None

    def _unknown_on_chord(self,position,heading,target,planner,boxes):
        observed=getattr(planner,'planning_observed',None)
        if observed is None:return False
        occupied=~planner.free & observed
        for box in boxes:occupied |= planner.box_mask(box)
        xx,yy=navigation_footprint_samples(planner);c,s=math.cos(heading),math.sin(heading)
        dx=np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        dy=np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        unknown=False
        for t in np.linspace(0,1,13):
            xy=np.asarray(position[:2])+t*(np.asarray(target[:2])-position[:2])
            row,col=planner.cell(xy);rows=row+dy;cols=col+dx
            if rows.min()<0 or cols.min()<0 or rows.max()>=observed.shape[0] or cols.max()>=observed.shape[1]:return False
            if occupied[rows,cols].any():return False
            unknown |= bool((~observed[rows,cols]).any())
        return unknown

    def _observe_gap(self,position,heading,target,planner,boxes,now,confirmed_gap=False):
        if not confirmed_gap and not self._unknown_on_chord(position,heading,target,planner,boxes):
            self._observation_started=None;return None
        if self._observation_started is None:
            self._observation_started=now;self._observation_yaw_origin=heading
            delta=np.asarray(target)-position[:2]
            error=(math.atan2(delta[1],delta[0])-heading+math.pi)%(2*math.pi)-math.pi
            self._observation_yaw_sign=1. if error>=0 else -1.
            self._observation_yaw_moving=False
        elapsed=now-self._observation_started
        if elapsed>8.:return None
        delta=np.asarray(target)-position[:2]
        progress=abs((heading-self._observation_yaw_origin+math.pi)%(2*math.pi)-math.pi)
        if abs(self.limiter.measured[2])>.15:self._observation_yaw_moving=True
        sign=self._observation_yaw_sign
        # From a settled stance the measured yaw response can be much lower
        # than the command. Use the calibrated1.0 request to initiate the
        # turn, then.6 after actual rotation starts. Keep the turn direction
        # fixed until enough sensor viewpoint change has been achieved.
        yaw_request=.6 if self._observation_yaw_moving else 1.
        candidates=([np.array([0.,0.,sign*yaw_request]),np.array([0.,0.,-sign*yaw_request])]
                    if elapsed<3. and progress<.35 else [])
        if not candidates and (abs(self.limiter.command[2])>.1 or abs(self.limiter.measured[2])>.15):
            return np.zeros(3)
        lateral_sign=1. if -math.sin(heading)*delta[0]+math.cos(heading)*delta[1]>=0 else -1.
        candidates.extend([np.array([0.,lateral_sign*.30,0.]),np.array([0.,-lateral_sign*.30,0.]),np.array([-.35,0.,0.])])
        for candidate in candidates:
            # Observation itself must fit entirely in already observed free
            # space. Never "fix" a blind band by declaring it free.
            if body_motion_clear(planner,boxes,position,heading,candidate[0],candidate[2],horizon=.8,lateral=candidate[1]):
                if abs(candidate[2])>.1:self._observation_yaw_sign=math.copysign(1.,candidate[2])
                return candidate
        return None

    def _tracking_segment_clear(self,start,target):
        # The global guide is a centerline on the inflated observed map.
        # The full oriented body is checked on its actual turning motion,
        # not translated at a fixed yaw all the way to a distant waypoint.
        return AStarMapPlanner.segment_clear(self._planner.cell(start),self._planner.cell(target),self._route_blocked)

    @staticmethod
    def _only_unknown_blocks(command,position,heading,planner,boxes,horizon):
        observed=getattr(planner,'planning_observed',None)
        if observed is None:return False
        saved=planner.free
        try:
            # Diagnostic classification only. No motion is accepted against
            # this temporary mask; observation movements use the real mask.
            planner.free=saved | ~observed
            return body_motion_clear(planner,boxes,position,heading,command[0],command[2],horizon=horizon,lateral=command[1])
        finally:planner.free=saved

    def action(self,position,heading,planner,boxes):
        now=float(getattr(planner,'control_time',0.))
        scope=getattr(planner,'planning_scope','local')
        try:
            planner.planning_scope='global';self._route_blocked=route_blocked(planner,boxes)
        finally:planner.planning_scope=scope
        # Baseline exploration's interpolated target and waypoint progression.
        action,state=super().action(position,heading,planner,boxes)
        delta=np.asarray(state['target'])-np.asarray(position[:2]);distance=float(np.linalg.norm(delta))
        error=(math.atan2(delta[1],delta[0])-heading+math.pi)%(2*math.pi)-math.pi
        remaining=float(np.linalg.norm(np.asarray(self._planned_path[-1])-position[:2]))
        factor=max(0.,math.cos(error))**2
        c,s=math.cos(heading),math.sin(heading)
        local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        desired=np.asarray(action['move_by_speed'],float).copy()
        if self.limiter.progress_position is None:
            self.limiter.progress_position=np.asarray(position[:2]).copy();self.limiter.progress_time=now
        elif np.linalg.norm(np.asarray(position[:2])-self.limiter.progress_position)>.05:
            self.limiter.progress_position=np.asarray(position[:2]).copy();self.limiter.progress_time=now
        stalled=now-self.limiter.progress_time>1.5
        target_heading=heading+error
        corner_heading=None
        index=self._waypoint_index
        if stalled and index+1<len(self._planned_path):
            corner=np.asarray(self._planned_path[index]);following=np.asarray(self._planned_path[index+1])
            to_corner=corner-np.asarray(position[:2]);segment=following-corner
            approach_error=(math.atan2(to_corner[1],to_corner[0])-heading+math.pi)%(2*math.pi)-math.pi
            future_heading=math.atan2(segment[1],segment[0])
            future_error=(future_heading-heading+math.pi)%(2*math.pi)-math.pi
            if (np.linalg.norm(to_corner)<.28 and np.linalg.norm(segment)>.08
                    and abs(approach_error)<.35 and abs(future_error)>.35):
                # A short fallback carrot can point straight at the near
                # waypoint while the body must turn for the outgoing segment.
                # Use that route tangent for a checked turn, never for an
                # unchecked shortcut across the corner.
                corner_heading=future_heading;target_heading=heading+future_error
                state['blocked_corner_heading_rad']=float(future_heading)
        stage_error=(target_heading-heading+math.pi)%(2*math.pi)-math.pi
        sharp_turn=not state.get('recovering_buffer') and abs(stage_error)>.85 and distance>.10
        if not self.staging_active and sharp_turn:
            # A world-directed shortcut can request reverse/strafe and large
            # yaw together when a frontier or corner moves behind the dog.
            # Brake first, then finish a checked turn before resuming travel.
            candidate=LocalStagingControl(position,heading,target_heading,planner,boxes,now)
            if not candidate.failed:
                candidate.phase='staging_brake'
                self.limiter.staging=candidate
            else:
                desired[:]=0.;state['needs_replan']=True
                state['sharp_turn_space_unavailable']=True
        elif (not self.staging_active and stalled and (not state.get('recovering_buffer') or corner_heading is not None) and abs(stage_error)>.25
                and np.linalg.norm(self.limiter.measured[:2])<.08
                and (corner_heading is not None or not body_rotation_clear(planner,boxes,position,heading,target_heading,margin=.025))):
            candidate=LocalStagingControl(position,heading,target_heading,planner,boxes,now)
            if not candidate.failed:self.limiter.staging=candidate
        staging_command=None
        if self.staging_active:
            result=self.limiter.staging.update(position,heading,planner,boxes,self.limiter,now)
            if result is None:
                self.limiter.staging=None;self.limiter.progress_time=now
                self.limiter.progress_position=np.asarray(position[:2]).copy()
                state.update(needs_replan=True,buffer_cleared_replan=True,staging_completed=True,recovering_buffer=False)
            else:
                staging_command,stage_state=result;state.update(stage_state)
                if self.limiter.staging.failed:
                    self.limiter.staging=None;state['blocked_ticks']=100
        if staging_command is not None:desired=staging_command
        # Restore the actual baseline velocity law, including its preserved
        # world direction when lateral speed is limited. Do not multiply a
        # second cosine-squared factor onto the reference.
        preview_horizon=.60+float(np.linalg.norm(desired[:2]))/(2*self.limiter.linear_acceleration)
        desired_clear=body_motion_clear(planner,boxes,position,heading,desired[0],desired[2],horizon=preview_horizon,lateral=desired[1])
        unknown_gap=(not desired_clear and self._only_unknown_blocks(desired,position,heading,planner,boxes,preview_horizon))
        if state.get('staging_completed'):
            # The staging maneuver displaced the base away from the old
            # route. Hand off to a route from this actual pose before moving.
            desired=np.zeros(3)
        elif staging_command is None and (state.get('needs_replan') or unknown_gap):
            observation=self._observe_gap(position,heading,state['target'],planner,boxes,now,confirmed_gap=unknown_gap)
            if observation is not None:
                desired=observation;state['needs_replan']=False
                state['observation_motion']=True;state['observation_reason']='unobserved_cells_in_body_corridor'
                self.limiter.progress_time=now
                self.limiter.progress_position=np.asarray(position[:2]).copy()
            else:
                desired[:]=0.;state['needs_replan']=True;state['observation_timeout_or_blocked']=True
        else:self._observation_started=None
        if state.get('recovering_buffer') and not state.get('local_staging'):
            # A short, continuously accelerated translation leaves mild map
            # overlap. Do not alternate large fixed sidesteps and yaw modes.
            direction=local/max(distance,1e-9)
            desired=np.array([*(.25*direction),0.])
        smoothed=self.limiter.preview(desired,now)
        if state.get('recovering_buffer') and not state.get('local_staging'):smoothed[2]=0.
        horizon=.60+max(float(np.linalg.norm(smoothed[:2])),float(np.linalg.norm(self.limiter.measured[:2])))/(2*self.limiter.linear_acceleration)
        clear=body_motion_clear(planner,boxes,position,heading,smoothed[0],smoothed[2],horizon=horizon,lateral=smoothed[1])
        if not clear and state.get('recovering_buffer'):
            world=np.array([c*smoothed[0]-s*smoothed[1],s*smoothed[0]+c*smoothed[1]])
            if abs(smoothed[2])<1e-8:
                clear=(body_translation_recovery_clear(planner,boxes,position,heading,state['target']) and
                    body_translation_recovery_clear(planner,boxes,position,heading,
                        np.asarray(position[:2])+world*max(horizon,.5),require_clear_end=False))
                if clear:state['overlap_recovery']=True
        if not clear:
            # Keep one continuous twist when possible. These alternatives are
            # all checked AFTER acceleration limiting, not just at the target.
            candidates=[]
            for scale in [.75,.40,0.]:
                for yaw in ([desired[2]] if state.get('local_staging') else [desired[2],0.,-.6,.6]):
                    target=np.array([desired[0]*scale,desired[1]*scale,yaw])
                    cmd=self.limiter.preview(target,now)
                    if body_motion_clear(planner,boxes,position,heading,cmd[0],cmd[2],horizon=horizon,lateral=cmd[1]):
                        cost=np.linalg.norm(cmd-desired)+.15*np.linalg.norm(cmd-self.limiter.command)
                        candidates.append((cost,cmd))
            if candidates:smoothed=min(candidates,key=lambda item:item[0])[1];clear=True
            else:
                # Emergency stop takes priority if a new scan reveals a
                # collision inside the braking distance. This is logged.
                smoothed=np.zeros(3);state['emergency_collision_stop']=True
        measured=self.limiter.measured
        if state.get('overlap_recovery'):
            measured_world=np.array([c*measured[0]-s*measured[1],s*measured[0]+c*measured[1]])
            measured_clear=body_translation_recovery_clear(planner,boxes,position,heading,np.asarray(position[:2])+measured_world*.25,require_clear_end=False)
        else:measured_clear=body_motion_clear(planner,boxes,position,heading,measured[0],measured[2],horizon=.25,lateral=measured[1])
        if (np.linalg.norm(measured[:2])>.10 or abs(measured[2])>.15) and not measured_clear:
            smoothed=np.zeros(3);clear=False;state['measured_braking_space_blocked']=True
        stopped_by_check=(not clear) or state.get('needs_replan',False) or (np.linalg.norm(smoothed)<.02 and np.linalg.norm(desired)>.15)
        self._blocked_ticks=self._blocked_ticks+1 if stopped_by_check else 0
        if self._blocked_ticks>30:state['needs_replan']=True
        self.limiter.accept(smoothed,now)
        state.update(controller=('continuous_'+state['staging_phase'] if state.get('local_staging') else 'continuous_exploration_velocity'),target=state['target'],
            desired_command=desired.tolist(),applied_command=smoothed.tolist(),
            heading_error_rad=error,body_sweep_clear=bool(clear and not state.get('overlap_recovery')),body_motion_safe=bool(clear),blocked_ticks=max(self._blocked_ticks,state.get('blocked_ticks',0)))
        return {'move_by_speed':smoothed.tolist()},state


def smooth_recovery_action(action,position,heading,planner,boxes,limiter,now):
    """Preserve a geometry-checked recovery's rotation versus translation."""
    raw=np.asarray(action['move_by_speed'],float)
    command=limiter.preview(raw,now)
    if abs(raw[2])>1e-8 and np.linalg.norm(raw[:2])<1e-8:
        if np.linalg.norm(limiter.measured[:2])>.10:return {'move_by_speed':[0.,0.,0.]}
        command[:2]=0.
        command[2]=math.copysign(min(abs(command[2]),abs(raw[2])),raw[2]) if command[2]*raw[2]>0 else 0.
        return {'move_by_speed':command.tolist()}
    command[2]=0.;c,s=math.cos(heading),math.sin(heading)
    world=np.array([c*command[0]-s*command[1],s*command[0]+c*command[1]])
    if body_translation_recovery_clear(planner,boxes,position,heading,np.asarray(position[:2])+world*.5,require_clear_end=False):
        return {'move_by_speed':command.tolist()}
    return {'move_by_speed':[0.,0.,0.]}
