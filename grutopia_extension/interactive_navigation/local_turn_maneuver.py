"""Find local turning space for a long robot instead of squeezing a circle route."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_rotation_clear,body_translation_recovery_clear,body_motion_clear)


def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi


def turn_placement_clear(planner,boxes,position,first_heading,last_heading,margin):
    """Require a half-cell pose neighborhood before committing to a stop.

    A millimeter grid crossing is insufficient: braking drift can put the
    base back in the previous cell and repeatedly restart translation.
    Check every base cell touched by the closed half-cell box. Sampling only
    its corners can lose a cell at a floating-point grid boundary, accepting
    a target whose actual settling neighborhood is not clear.
    """
    position=np.asarray(position[:2],float)
    radius=.5*planner.resolution
    origin=np.array([planner.limits[0][0],planner.limits[1][0]])
    # Numerical tolerance in cell units only (5e-11m for a 5cm map).
    lower=np.floor((position-radius-origin)/planner.resolution-1e-9).astype(int)
    upper=np.floor((position+radius-origin)/planner.resolution+1e-9).astype(int)
    for row in range(lower[1],upper[1]+1):
        for col in range(lower[0],upper[0]+1):
            if not planner.valid_cell((row,col)):return False
            xy=origin+(np.array([col,row])+.5)*planner.resolution
            if not body_rotation_clear(planner,boxes,xy,first_heading,last_heading,margin=margin):return False
    return True


def prepare_turn_plan(planner,boxes,position,heading,goal_heading):
    """Plan an in-place preparation before a checked short translation.

    Some pockets cannot translate at the current heading or turn directly
    to the route. Keep the .8m search bound and original margins, but check
    a preparation turn, translation, and final turn as separate motions.
    Screen the translation/final turn at nominal and +/- .12rad finish
    headings; execution must recheck the actual handoff pose.
    """
    from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear
    position=np.asarray(position[:2],float);row,col=planner.cell(position)
    reach=int(math.ceil(.8/planner.resolution))+1;targets=[]
    for rr in range(max(0,row-reach),min(planner.free.shape[0],row+reach+1)):
        for cc in range(max(0,col-reach),min(planner.free.shape[1],col+reach+1)):
            if not planner.free[rr,cc]:continue
            target=np.array([planner.limits[0][0]+(cc+.5)*planner.resolution,
                             planner.limits[1][0]+(rr+.5)*planner.resolution])
            distance=float(np.linalg.norm(target-position))
            if .02<distance<=.8:targets.append((distance,target))
    targets.sort(key=lambda item:item[0])
    angles=[goal_heading+math.pi,*np.arange(-math.pi,math.pi,math.pi/12)]
    angles=sorted({round(heading+wrap(a-heading),10) for a in angles},key=lambda a:abs(a-heading))
    for margin in [.05,.025]:
        for prepared in angles:
            if not .25<=abs(prepared-heading)<=1.5:continue
            if not turn_placement_clear(planner,boxes,position,heading,prepared,margin):continue
            if not stepping_arc_clear(planner,boxes,position,heading,math.copysign(1.,prepared-heading)):continue
            for distance,target in targets:
                valid=True
                for offset in [0.,-.12,.12]:
                    actual=prepared+offset;end=actual+wrap(goal_heading-actual)
                    if not (body_translation_recovery_clear(planner,boxes,position,actual,target)
                            and turn_placement_clear(planner,boxes,target,actual,end,margin)
                            and stepping_arc_clear(planner,boxes,target,actual,math.copysign(1.,end-actual))):
                        valid=False;break
                if valid:return {'prepared_heading':prepared,'target':target,'margin':margin,
                                 'goal_heading':float(goal_heading),'translation_m':distance}
    return None


class LocalTurnManeuver:
    def __init__(self,goal_heading,max_ticks=2500):
        self.goal_heading=float(goal_heading)
        self.stage_target=None;self.stage_margin=.05
        self.reserve_margin=0.
        self.ticks=0;self.max_ticks=max_ticks;self.turn_ticks=0;self.turn_sign=0.
        self.origin=None;self.failed=False

    def _select_stage(self,planner,boxes,position,heading,require_stepping_arc=False,require_stable_placement=False):
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
        def select(candidates):
            for margin in [.05,.025]:
                for _,target in candidates:
                    cell=planner.cell(target)
                    if not planner.valid_cell(cell) or not planner.free[cell]:continue
                    if require_stepping_arc:
                        from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear
                        if not stepping_arc_clear(planner,boxes,target,heading,math.copysign(1.,last_heading-heading)):continue
                    if (body_rotation_clear(planner,boxes,target,heading,last_heading,margin=margin+self.reserve_margin)
                            and body_translation_recovery_clear(planner,boxes,position,heading,target)):
                        if require_stable_placement and not turn_placement_clear(planner,boxes,target,heading,last_heading,margin+self.reserve_margin):continue
                        self.stage_margin=margin
                        return target
            return None
        target=select(choices)
        if target is not None:return target
        # A 15cm body-aligned lattice can miss a safe pocket only 10cm away
        # (recorded qualified T02). Search map-cell centers as a fallback,
        # with the same .8m reach and every original motion/placement check.
        # This samples poses; it does not clear unknown cells or shrink the
        # body, stepping arc, margin, or staging execution budget.
        fine=[];row,col=planner.cell(position);reach=int(math.ceil(.8/planner.resolution))+1
        for rr in range(max(0,row-reach),min(planner.free.shape[0],row+reach+1)):
            for cc in range(max(0,col-reach),min(planner.free.shape[1],col+reach+1)):
                if not planner.free[rr,cc]:continue
                target=np.array([planner.limits[0][0]+(cc+.5)*planner.resolution,
                                 planner.limits[1][0]+(rr+.5)*planner.resolution])
                delta=target-position;distance=float(np.linalg.norm(delta))
                if distance>.8:continue
                forward=c*delta[0]+s*delta[1]
                fine.append((distance+.15*max(forward,0.),target))
        fine.sort(key=lambda entry:entry[0])
        return select(fine)

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
