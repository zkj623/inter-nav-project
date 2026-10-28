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
        self.used_views={};self.attempts={};self.reviews={}

    def review(self,object_id,category,now):
        self.reviews.setdefault(object_id,[]).append((str(category),float(now)))

    def needs_inspection(self,object_id,recognitions,eligible_ids,now):
        if object_id in eligible_ids:return False
        item=recognitions.confirmed(object_id,now)
        if item is None:return True
        # An unsupported passive classification is checked from two active
        # views before it prevents further observation. This never grants a skill.
        reviews=self.reviews.get(object_id,[])
        return sum(category==item.category and now-observed<=60. for category,observed in reviews)<2

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
            if candidate.kind=='push':
                candidates=[b for b in boxes if b.id==candidate.push.box_id]
            else:
                # Looking at a nearby obstruction does not require knowing
                # the room beyond it, and never authorizes moving that object.
                start_xy=np.asarray(start[:2]);goal_delta=np.asarray(goal[:2])-start_xy
                length=np.linalg.norm(goal_delta)
                if length<.1:return None
                forward=goal_delta/length;candidates=[]
                for obj in boxes:
                    delta=np.asarray(obj.xy)-start_xy;along=float(np.dot(delta,forward))
                    lateral=float(np.linalg.norm(delta-along*forward))
                    if 0.<along<length+.3 and np.linalg.norm(delta)<=3.5 and lateral<1.2:candidates.append(obj)
                candidates.sort(key=lambda obj:np.linalg.norm(np.asarray(obj.xy)-start_xy))
            box=next((obj for obj in candidates if self.needs_inspection(obj.id,recognitions,p.interactable_ids,now)
                      and self.attempts.get(obj.id,0)<4),None)
            if box is None:return None
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
                                  'unclassified_obstacle_requires_rgb_view')
