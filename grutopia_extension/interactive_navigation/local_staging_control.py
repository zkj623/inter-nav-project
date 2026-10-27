"""Measured settling and continuous translation into a known safe turn pose."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear,body_translation_recovery_clear
from grutopia_extension.interactive_navigation.go2_turn_response import TurnResponseAssist,stepping_arc_clear


def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


class LocalStagingControl:
    def __init__(self,position,heading,goal_heading,planner,boxes,now):
        self.origin=np.asarray(position[:2],float).copy();self.started=now
        self.goal_heading=heading+wrap(goal_heading-heading)
        self.selector=LocalTurnManeuver(self.goal_heading)
        self.target=self.selector._select_stage(planner,boxes,self.origin,heading,require_stepping_arc=True)
        self.phase='staging_move';self.stable_since=None;self.failed=self.target is None
        self.turn_response=TurnResponseAssist()

    def update(self,position,heading,planner,boxes,limiter,now):
        position=np.asarray(position[:2],float)
        state={'local_staging':True,'target':position.tolist(),'target_heading_rad':self.goal_heading,
               'staging_phase':self.phase,'recovering_buffer':True,'needs_replan':False}
        if self.failed or now-self.started>20. or np.linalg.norm(position-self.origin)>1.2:
            self.failed=True
            return np.zeros(3),{**state,'needs_replan':True,'staging_phase':'staging_failed'}
        turn_end=heading+wrap(self.goal_heading-heading)
        can_turn=body_rotation_clear(planner,boxes,position,heading,turn_end,margin=.025)
        can_start_turn=can_turn and stepping_arc_clear(planner,boxes,position,heading,math.copysign(1.,turn_end-heading))
        settled=(np.linalg.norm(limiter.measured[:2])<.06 and abs(limiter.measured[2])<.10
                 and np.linalg.norm(limiter.command[:2])<.06 and abs(limiter.command[2])<.10)
        error=wrap(self.goal_heading-heading)
        if self.phase=='staging_move':
            if can_start_turn:
                self.phase='staging_brake';self.stable_since=None
            else:
                if (not body_rotation_clear(planner,boxes,self.target,heading,turn_end,margin=.025)
                        or not stepping_arc_clear(planner,boxes,self.target,heading,math.copysign(1.,turn_end-heading))
                        or not body_translation_recovery_clear(planner,boxes,position,heading,self.target)):
                    self.target=self.selector._select_stage(planner,boxes,position,heading,require_stepping_arc=True)
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
                speed=float(np.clip(1.5*distance-.6*toward,0.,.30))
                world=direction*speed
                command=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],0.])
                return command,{**state,'target':self.target.tolist(),'staging_phase':self.phase}
        if self.phase in ['staging_brake','staging_turn_brake']:
            if settled:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.20:
                    if self.phase=='staging_turn_brake' and abs(error)<.12:return None
                    self.phase='staging_turn' if can_start_turn else 'staging_move'
                    self.stable_since=None
            else:self.stable_since=None
            return np.zeros(3),{**state,'staging_phase':self.phase}
        if self.phase=='staging_turn':
            if not can_turn:
                self.phase='staging_brake';self.stable_since=None
                return np.zeros(3),{**state,'staging_phase':self.phase}
            stopping_angle=abs(limiter.measured[2])*.25+limiter.command[2]**2/(2*limiter.yaw_acceleration)+.03
            if (abs(error)<.08 or (error*limiter.measured[2]>0 and abs(limiter.measured[2])>.1 and abs(error)<stopping_angle)
                    or (error*limiter.measured[2]<0 and abs(limiter.measured[2])>.1)):
                self.phase='staging_turn_brake';self.stable_since=None
                return np.zeros(3),{**state,'staging_phase':self.phase}
            request=np.array([0.,0.,math.copysign(1.,error)])
            request,detail=self.turn_response.update(request,position,heading,planner,boxes,limiter,now)
            if detail.get('turn_assist')=='no_observed_safe_arc':
                self.phase='staging_move';self.turn_response=TurnResponseAssist()
                request=np.zeros(3)
            return request,{**state,'staging_phase':self.phase,**detail}
        return np.zeros(3),{**state,'staging_phase':self.phase}
