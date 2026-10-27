"""Continuous exploration velocity tracking with acceleration-bounded requests.

The original Go2 policy/actuator loop is unchanged. Local checks evaluate the
command actually sent after smoothing, including the full body footprint.
"""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear,body_translation_recovery_clear,navigation_body_blocked
from scipy.ndimage import distance_transform_edt


class VelocitySlewLimiter:
    def __init__(self,linear_acceleration=.65,yaw_acceleration=1.8):
        self.linear_acceleration=linear_acceleration;self.yaw_acceleration=yaw_acceleration
        self.command=np.zeros(3);self.last_time=None
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
        super().__init__(start,path,lookahead=.55)
        self.limiter=limiter;self.max_forward=max_forward;self.max_lateral=max_lateral
        self._last_plan_time=-1e9;self._desired=np.zeros(3)
        self._blocked_ticks=0;self._last_safe=np.zeros(3)

    def action(self,position,heading,planner,boxes):
        now=float(getattr(planner,'control_time',0.))
        # Baseline exploration's interpolated target and waypoint progression.
        action,state=super().action(position,heading,planner,boxes)
        delta=np.asarray(state['target'])-np.asarray(position[:2]);distance=float(np.linalg.norm(delta))
        error=(math.atan2(delta[1],delta[0])-heading+math.pi)%(2*math.pi)-math.pi
        remaining=float(np.linalg.norm(np.asarray(self._planned_path[-1])-position[:2]))
        factor=max(0.,math.cos(error))**2
        c,s=math.cos(heading),math.sin(heading)
        local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        desired=np.array([self.max_forward*factor*min(1.,remaining/.50),
            np.clip(1.2*local[1],-self.max_lateral,self.max_lateral)*factor,
            np.clip(1.8*error,-1.2,1.2)])
        if state.get('needs_replan'):desired[:]=0.
        if state.get('recovering_buffer'):
            # A short, continuously accelerated translation leaves mild map
            # overlap. Do not alternate large fixed sidesteps and yaw modes.
            direction=local/max(distance,1e-9)
            desired=np.array([*(.25*direction),0.])
        smoothed=self.limiter.preview(desired,now)
        horizon=.35+float(np.linalg.norm(smoothed[:2]))/(2*self.limiter.linear_acceleration)
        clear=body_motion_clear(planner,boxes,position,heading,smoothed[0],smoothed[2],horizon=horizon,lateral=smoothed[1])
        if not clear and state.get('recovering_buffer'):
            world=np.array([c*smoothed[0]-s*smoothed[1],s*smoothed[0]+c*smoothed[1]])
            clear=body_translation_recovery_clear(planner,boxes,position,heading,
                np.asarray(position[:2])+world*max(horizon,.5))
        if not clear:
            # Keep one continuous twist when possible. These alternatives are
            # all checked AFTER acceleration limiting, not just at the target.
            candidates=[]
            for scale in [.75,.40,0.]:
                for yaw in [desired[2],0.,-.6,.6]:
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
        stopped_by_check=(not clear) or (np.linalg.norm(smoothed)<.02 and np.linalg.norm(desired)>.15)
        self._blocked_ticks=self._blocked_ticks+1 if stopped_by_check else 0
        if self._blocked_ticks>30:state['needs_replan']=True
        self.limiter.accept(smoothed,now)
        state.update(controller='continuous_exploration_velocity',target=state['target'],
            desired_command=desired.tolist(),applied_command=smoothed.tolist(),
            heading_error_rad=error,body_sweep_clear=bool(clear),blocked_ticks=self._blocked_ticks)
        return {'move_by_speed':smoothed.tolist()},state
