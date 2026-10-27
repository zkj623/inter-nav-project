"""Find local turning space for a long robot instead of squeezing a circle route."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_rotation_clear,body_translation_recovery_clear,body_motion_clear)


def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi


class LocalTurnManeuver:
    def __init__(self,goal_heading,max_ticks=2500):
        self.goal_heading=float(goal_heading)
        self.stage_target=None;self.stage_margin=.05
        self.ticks=0;self.max_ticks=max_ticks;self.turn_ticks=0;self.turn_sign=0.
        self.origin=None;self.failed=False

    def _select_stage(self,planner,boxes,position,heading,require_stepping_arc=False):
        c,s=math.cos(heading),math.sin(heading)
        last_heading=heading+wrap(self.goal_heading-heading)
        choices=[]
        for forward in [0.,-.15,.15,-.30,.30,-.45,.45,-.60,.60]:
            for lateral in [0.,-.15,.15,-.30,.30,-.45,.45,-.60,.60]:
                distance=math.hypot(forward,lateral)
                if not .08<distance<=.8:continue
                target=position+np.array([c*forward-s*lateral,s*forward+c*lateral])
                choices.append((distance+.15*max(forward,0.),target))
        choices.sort(key=lambda entry:entry[0])
        for margin in [.05,.025]:
            for _,target in choices:
                cell=planner.cell(target)
                if not planner.valid_cell(cell):continue
                if require_stepping_arc:
                    from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear
                    if not stepping_arc_clear(planner,boxes,target,heading,math.copysign(1.,last_heading-heading)):continue
                if (body_rotation_clear(planner,boxes,target,heading,last_heading,margin=margin)
                        and body_translation_recovery_clear(planner,boxes,position,heading,target)):
                    self.stage_margin=margin
                    return target
        return None

    def action(self,position,heading,planner,boxes):
        position=np.asarray(position[:2],float)
        if self.origin is None:self.origin=position.copy()
        self.ticks+=1
        error=wrap(self.goal_heading-heading)
        if abs(error)<.08:return None
        state={'target':position.tolist(),'recovering_buffer':True,'needs_replan':False,
            'target_heading_rad':self.goal_heading,'local_turn_maneuver':True}
        if self.ticks>self.max_ticks or np.linalg.norm(position-self.origin)>1.2:
            self.failed=True
            return {'move_by_speed':[0.,0.,0.]},{**state,'needs_replan':True,'controller':'local_turn_staging_timeout'}
        if body_rotation_clear(planner,boxes,position,heading,heading+error,margin=.025):
            sign=math.copysign(1.,error)
            if sign!=self.turn_sign:self.turn_ticks=0
            self.turn_sign=sign
            rotation=sign*min(1.,.4+.01*self.turn_ticks)
            for candidate in [rotation,sign*.6,sign*.4]:
                if body_motion_clear(planner,boxes,position,heading,0.,candidate):
                    self.turn_ticks+=1
                    return {'move_by_speed':[0.,0.,candidate]},{**state,'controller':'local_align_to_path'}
        self.turn_ticks=0
        if self.stage_target is None or not body_rotation_clear(planner,boxes,self.stage_target,heading,heading+error,margin=self.stage_margin):
            self.stage_target=self._select_stage(planner,boxes,position,heading)
        if self.stage_target is None:
            self.failed=True
            return {'move_by_speed':[0.,0.,0.]},{**state,'needs_replan':True,'controller':'no_local_turning_pose'}
        delta=self.stage_target-position;c,s=math.cos(heading),math.sin(heading)
        body_delta=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        order=[0,1] if body_delta[0]<-.04 else list(np.argsort(-np.abs(body_delta)))
        for axis in order:
            if abs(body_delta[axis])<.015:continue
            command=np.zeros(3)
            command[axis]=math.copysign(.35 if axis==0 and body_delta[0]<0 else .30,body_delta[axis])
            world=np.array([c*command[0]-s*command[1],s*command[0]+c*command[1]])
            target=position+world*.25
            if body_translation_recovery_clear(planner,boxes,position,heading,target):
                return {'move_by_speed':command.tolist()},{**state,'target':self.stage_target.tolist(),'controller':'local_turn_staging'}
        self.stage_target=None
        return {'move_by_speed':[0.,0.,0.]},{**state,'needs_replan':True,'controller':'turn_staging_motion_blocked'}
