"""Run with the documented RSI checkout on PYTHONPATH; no simulator required."""
import json,math
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import distance_transform_edt
from g2_pose_planning import PoseGoalPlanner,collision_mask
from g2_voronoi_guidance import RSIVoronoiGuidance
from grutopia_extension.interactive_navigation.mapping import PlanningError

limits=[[-15,15],[-15,15]];rr,cc=np.indices((600,600));x=-15+(cc+.5)*.05;y=-15+(rr+.5)*.05
def layers(free):return SimpleNamespace(observed=np.ones_like(free),robot_footprint=np.zeros_like(free),local_occupied=~free,global_occupied=~free)
free=(abs(x)<3)&(abs(y)<1.5);planner=PoseGoalPlanner(limits);route=planner.plan(layers(free),np.array([-2.,1.05]),0.,np.array([2.,1.05]),0.);assert route
record=planner.guidance_records[-1];assert record['engine']=='rsi_connected_clearance_weighted_voronoi';assert record['implementation'].endswith('ConnectedVoronoiPathPlanner');field=distance_transform_edt(free)*.05;gap=float(np.median([field[planner.cell(p)] for p,a in route if abs(p[0])<1.]));assert gap>1.,gap
reference=np.array(record['reference_xy_m']);k=0
for p,a in route:
 while k<len(reference) and np.linalg.norm(p-reference[k])<=.075+1e-8:k+=1
assert k==len(reference)
known=np.ones_like(free);known[:,295:305]=False
try:RSIVoronoiGuidance(limits,.05).reference(free&known,known,np.array([-2.,0.]),np.array([2.,0.]))
except PlanningError:unknown_blocked=True
else:unknown_blocked=False
assert unknown_blocked

# A measured-center obstacle that blocks rounding yaw to pi but does not
# block the measured pose. The planner must attach using an actually safe move.
planner=PoseGoalPlanner(limits);p=np.array([.025,.025])+[.0083846092,.0011066198];yaw=3.0427894307;r,c=planner.cell(p);offset=p-planner.xy(r,c);angles=np.linspace(yaw,math.pi,10)
grid=np.logical_or.reduce([collision_mask(float(a),.05,.05) for a in angles]);actual=np.logical_or.reduce([collision_mask(float(a),.05,.05,*offset) for a in angles]);standing=collision_mask(yaw,.05,.025,*offset);candidates=np.argwhere(actual&~grid&~standing);assert len(candidates)
free=(abs(x)<3)&(abs(y)<1.5);dy,dx=candidates[0]-13;free[r+dy,c+dx]=False;state=layers(free);planner.route=planner.plan(state,p,yaw,np.array([-1.,.6]),0.);assert planner.route;cmd=planner.command(state,p,yaw,np.array([-1.,.6]),.1);assert cmd is not None and np.linalg.norm(cmd)>0 and planner.status!='rotation_sweep_not_observed_free';assert planner.guidance_records[-1]['measured_heading_attachment']
print(json.dumps({'passed':True,'original_rsi_engine':record['implementation'],'corridor_interior_center_clearance_m':gap,'ordered_reference_waypoints':len(reference),'unknown_cells_block_both_engines':unknown_blocked,'measured_heading_attachment_regression':True,'measured_pose_first_command':cmd.tolist()},indent=2))
