"""Collision-checked lookahead tracking with body-frame gait commands."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


class Go2VelocityPathTracker:
    def __init__(self,start,path,lookahead=.35,fixed_heading=None):
        self.points=np.array([tuple(start),*[tuple(p[:2]) for p in path]],float)
        self.lookahead=lookahead;self.cursor=0
        self._blocked=None;self._calls=0
        self.fixed_heading=fixed_heading
        self._projection=None;self._normal=None;self._correcting=False;self._aligning=False

    def target(self,position,planner,blocked):
        p=np.array(position[:2]);cell=planner.cell(p)
        if planner.valid_cell(cell) and blocked[cell]:
            safe=distance_transform_edt(~blocked)*planner.resolution>.12
            if not safe.any():safe=~blocked
            _,nearest=distance_transform_edt(~safe,return_indices=True)
            rc=nearest[:,cell[0],cell[1]]
            target=np.array([planner.limits[0][0]+(rc[1]+.5)*planner.resolution,
                             planner.limits[1][0]+(rc[0]+.5)*planner.resolution])
            return target,True
        best=None
        for i in range(max(0,self.cursor-1),len(self.points)-1):
            a,b=self.points[i:i+2];segment=b-a
            t=np.clip(np.dot(p-a,segment)/max(np.dot(segment,segment),1e-10),0,1)
            projected=a+t*segment;distance=np.linalg.norm(p-projected)
            if best is None or distance<best[0]:best=(distance,i,projected)
        if best is None:return self.points[-1],False
        _,index,projection=best;self.cursor=index
        self._projection=projection.copy()
        segment=self.points[index+1]-self.points[index]
        length=np.linalg.norm(segment)
        self._normal=np.array([-segment[1],segment[0]])/length if length>1e-8 else None
        for length in [self.lookahead,.2,.1,.05,.02]:
            target=projection.copy();remaining=length;segment_index=index
            while segment_index<len(self.points)-1:
                end=self.points[segment_index+1];delta=end-target;distance=np.linalg.norm(delta)
                if distance>=remaining:
                    if distance>1e-8:target+=delta*(remaining/distance)
                    break
                remaining-=distance;target=end.copy();segment_index+=1
            if AStarMapPlanner.segment_clear(planner.cell(p),planner.cell(target),blocked):return target,False
        next_vertex=self.points[min(index+1,len(self.points)-1)]
        if np.linalg.norm(next_vertex-p)>.003 and AStarMapPlanner.segment_clear(planner.cell(p),planner.cell(next_vertex),blocked):
            return next_vertex,False
        return projection,True

    def action(self,position,heading,planner,boxes):
        if self._blocked is None or self._calls%20==0:self._blocked=planner.blocked(boxes)
        self._calls+=1;blocked=self._blocked
        target,recovering=self.target(position,planner,blocked)
        error=target-np.array(position[:2]);distance=np.linalg.norm(error)
        maximum=.30 if recovering else .35
        velocity=.9*error
        norm=np.linalg.norm(velocity)
        minimum=.30 if recovering else .10
        if self.fixed_heading is not None and distance>.005 and norm<minimum:
            velocity*=minimum/max(norm,1e-8);norm=np.linalg.norm(velocity)
        if norm>maximum:velocity*=maximum/norm
        cross_track=0.
        if self.fixed_heading is not None and not recovering and self._normal is not None:
            cross_track=float(np.dot(np.array(position[:2])-self._projection,self._normal))
            if abs(cross_track)>.035:self._correcting=True
            if abs(cross_track)<.012:self._correcting=False
            if self._correcting:
                correction=-math.copysign(.30,cross_track)*self._normal
                end=np.array(position[:2])+correction*.10
                if AStarMapPlanner.segment_clear(planner.cell(position),planner.cell(end),blocked):
                    velocity=correction
        else:self._correcting=False
        if distance>.03 and not recovering:
            desired=math.atan2(error[1],error[0]) if self.fixed_heading is None else self.fixed_heading
            angle=(desired-heading+math.pi)%(2*math.pi)-math.pi
            angular=float(np.clip(.7*angle,-.25,.25))
        else:angular=0.
        if self.fixed_heading is not None and not recovering:
            angle=(self.fixed_heading-heading+math.pi)%(2*math.pi)-math.pi
            if abs(angle)>.07:self._aligning=True
            if abs(angle)<.025:self._aligning=False
            if self._aligning:
                velocity=np.zeros(2);angular=math.copysign(.25,angle)
        c,s=math.cos(heading),math.sin(heading)
        forward=float(np.clip(c*velocity[0]+s*velocity[1],-.25,.35))
        lateral_limit=.30 if self._correcting or recovering else .25
        lateral=float(np.clip(-s*velocity[0]+c*velocity[1],-lateral_limit,lateral_limit))
        return {'move_by_speed':[forward,lateral,angular]}, {'target':target.tolist(),'recovering_buffer':recovering,
            'cross_track_error_m':cross_track,'correcting_drift':self._correcting,'aligning_heading':self._aligning}
