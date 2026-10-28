"""Active RGB inspection of unclassified geometric obstacles on a blocked route."""
from dataclasses import dataclass
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


@dataclass
class InspectionPlan:
    object_id: str
    position: tuple
    heading: float
    reason: str


class ObjectInspectionPlanner:
    def __init__(self):
        self.used_views={};self.attempts={}

    def propose(self,layers,start,heading,goal,recognitions,now):
        p=layers.planner
        with layers.global_planning() as boxes:
            # A counterfactual selects what to LOOK AT, never what to push.
            saved=p.interactable_ids;passages=dict(p.active_passages);saved_heading=p.robot_heading
            try:
                p.robot_heading=None
                if p.path(start,goal,boxes):return None
                p.interactable_ids={b.id for b in boxes}
                candidate=p.decide(start,goal,boxes)
            finally:
                p.interactable_ids=saved;p.active_passages=passages;p.robot_heading=saved_heading
            if candidate.kind!='push':return None
            box=next(b for b in boxes if b.id==candidate.push.box_id)
            if recognitions.confirmed(box.id,now) is not None:return None
            if self.attempts.get(box.id,0)>=4:return None
            center=np.asarray(box.xy);position=np.asarray(start[:2]);toward=position-center
            theta=math.atan2(toward[1],toward[0]);choices=[]
            p.robot_heading=None
            try:
                for radius in [2.,2.4,1.7,2.8,1.5]:
                    for offset in [0.,-.25,.25,-.5,.5,-.8,.8,-1.05,1.05,-1.3,1.3,-1.6,1.6,-1.9,1.9,-2.2,2.2,-2.6,2.6,math.pi]:
                        point=center+radius*np.array([math.cos(theta+offset),math.sin(theta+offset)])
                        if any(np.linalg.norm(point-old)<.30 for old in self.used_views.get(box.id,[])):continue
                        view_heading=math.atan2(center[1]-point[1],center[0]-point[0])
                        if not body_rotation_clear(p,boxes,point,view_heading,view_heading,margin=.05):continue
                        path=p.path(position,point,boxes)
                        if not path:continue
                        arrival=heading if np.linalg.norm(position-point)<.15 else math.atan2(point[1]-path[-2][1],point[0]-path[-2][0]) if len(path)>1 else math.atan2(point[1]-position[1],point[0]-position[0])
                        if not body_rotation_clear(p,boxes,point,arrival,view_heading,margin=.05):continue
                        choices.append((p.length(position,path)+.3*abs(radius-2.2),point,view_heading))
            finally:p.robot_heading=saved_heading
            if not choices:return None
            _,point,view_heading=min(choices,key=lambda x:x[0])
            self.used_views.setdefault(box.id,[]).append(point.copy())
            self.attempts[box.id]=self.attempts.get(box.id,0)+1
            return InspectionPlan(box.id,tuple(map(float,point)),float(view_heading),
                                  'unclassified_obstacle_blocks_observed_goal_route')
