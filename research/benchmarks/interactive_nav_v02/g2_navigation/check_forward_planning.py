"""Behavior checks for forward preference and required lateral passage."""
import json,math
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from g2_pose_planning import PoseGoalPlanner,collision_mask

def check(half_width):
 limits=[[-15,15],[-15,15]];yy,xx=np.indices((600,600));x=-15+(xx+.5)*.05;y=-15+(yy+.5)*.05
 occupied=(abs(x)>2.5)|(abs(y)>half_width)
 layers=SimpleNamespace(observed=np.ones((600,600),bool),robot_footprint=np.zeros((600,600),bool),local_occupied=occupied,global_occupied=occupied)
 planner=PoseGoalPlanner(limits);route=planner.plan(layers,np.array([0.,0.]),math.pi/2,np.array([1.6,0.]),0.)
 assert route and planner.route_kind=='navigate',planner.status
 forward=lateral=reverse=turn=0.
 for (p,a),(q,b) in zip(route,route[1:]):
  d=q-p;n=np.linalg.norm(d)
  if n<1e-9:
   angle=(b-a+math.pi)%(2*math.pi)-math.pi;turn+=abs(angle);r,c=planner.cell(p);offset=p-planner.xy(r,c);patch=(~occupied)[r-13:r+14,c-13:c+14]
   for heading in np.linspace(a,a+angle,max(2,int(abs(angle)/(math.pi/64))+2)):
    assert np.all(patch[collision_mask(float(heading),.05,.05,float(offset[0]),float(offset[1]))]),'Turn violated measured-center full-body sweep'
   continue
  cosine=float(np.dot(d/n,[math.cos(a),math.sin(a)]))
  if cosine>math.cos(math.pi/6):forward+=n
  elif cosine<0:reverse+=n
  else:lateral+=n
 return dict(forward_m=forward,lateral_m=lateral,reverse_m=reverse,turn_rad=turn)
open_room=check(2.5);corridor=check(.45)
assert open_room['forward_m']>1.0 and open_room['lateral_m']<.1 and open_room['turn_rad']>1.,open_room
assert corridor['lateral_m']+corridor['reverse_m']>1.0 and corridor['turn_rad']<math.pi/4,corridor
layers=SimpleNamespace(observed=np.ones((600,600),bool),robot_footprint=np.zeros((600,600),bool),local_occupied=np.zeros((600,600),bool),global_occupied=np.zeros((600,600),bool))
planner=PoseGoalPlanner([[-15,15],[-15,15]]);planner.route_kind='navigate';planner.route=[(np.array([x,0.]),0.) for x in (0.,.05,.10,.15,.20)]
command=planner.command(layers,np.zeros(2),.033,np.array([2.,0.]),0.)
assert command[0]>.10 and abs(command[1])<.01 and planner.motion_mode=='forward',command
planner=PoseGoalPlanner([[-15,15],[-15,15]]);planner.route_kind='navigate';planner.route=[(np.zeros(2),math.pi/2)]
turn_command=planner.command(layers,np.zeros(2),0.,np.array([2.,0.]),0.)
assert np.allclose(turn_command[:2],0) and turn_command[2]>0 and planner.motion_mode=='turn_in_place',turn_command
planner=PoseGoalPlanner([[-15,15],[-15,15]]);planner.route_kind='navigate';planner.route=[(np.zeros(2),math.pi/2)]
offset_turn=planner.command(layers,np.array([.0201,0.]),0.,np.array([2.,0.]),0.)
assert np.allclose(offset_turn[:2],0) and planner.motion_mode=='turn_in_place','Unnecessary near-corner centering'
planner=PoseGoalPlanner([[-15,15],[-15,15]]);planner.route_kind='navigate';planner.route=[(np.zeros(2),math.pi/2)]
center_command=planner.command(layers,np.array([.06,0.]),0.,np.array([2.,0.]),0.)
assert np.linalg.norm(center_command[:2])>=.045 and planner.motion_mode=='align_rotation_center','Centering command too small'
yy,xx=np.indices((600,600));occupied=(abs(-15+(xx+.5)*.05)>2.5)|(abs(-15+(yy+.5)*.05)>.45)
layers.local_occupied=occupied;layers.global_occupied=occupied
planner=PoseGoalPlanner([[-15,15],[-15,15]]);planner.route_kind='navigate';planner.route=[(np.zeros(2),0.)]
blocked_command=planner.command(layers,np.zeros(2),math.pi/2,np.array([2.,0.]),0.)
assert np.allclose(blocked_command,0) and planner.status=='rotation_sweep_not_observed_free',planner.status
x=-15+(xx+.5)*.05;y=-15+(yy+.5)*.05
free=((x>-.6)&(x<2.4)&(abs(y)<.40))|((abs(x)<.4)&(y>=0)&(y<2.5))|((abs(x-1.8)<.6)&(abs(y)<.6))|((abs(x)<.6)&(abs(y)<.6))
layers.local_occupied=~free;layers.global_occupied=~free
planner=PoseGoalPlanner([[-15,15],[-15,15]]);route=planner.plan(layers,np.array([.7,0.]),0.,np.array([0.,1.8]),0.)
assert route and planner.route_kind=='navigate',planner.status
first=next((b[0]-a[0] for a,b in zip(route,route[1:]) if np.linalg.norm(b[0]-a[0])>.001),None)
assert first is not None and first[0]<0 and max(p[0] for p,a in route)<1.,'Unnecessary drive to distant turnaround bay'
print(json.dumps({'passed':True,'open_room':open_room,'narrow_corridor':corridor,'small_heading_error_forward_command':command.tolist(),'turn_command':turn_command.tolist(),'blocked_sweep_command':blocked_command.tolist(),'short_reverse_instead_of_distant_turnaround':True,'safe_offset_turn_command':offset_turn.tolist(),'effective_centering_command':center_command.tolist()},indent=2))
