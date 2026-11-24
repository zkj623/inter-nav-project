"""Observed-only short straight exit from an over-conservative lattice start.

Uses the union of the robot's actual native convex-part XY projections, not
scene geometry. Never erases occupancy or changes task poses. The ordinary
planner envelope remains required at the recovery endpoint.
"""
import json,math
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from shapely.geometry import Polygon
from shapely.ops import unary_union
try:
 from shapely import contains_xy
except ImportError:
 from shapely.vectorized import contains as contains_xy

def prepare_footprint(out):
 from pxr import Usd,UsdGeom,UsdPhysics
 out=Path(out);task=json.loads((out/'task.json').read_text());s=Usd.Stage.Open(str(out/'robot.usdc'));cache=UsdGeom.XformCache();polys=[];yaw=task['start_yaw_rad'];c,si=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-si],[si,c]])
 for p in s.Traverse():
  if not str(p.GetPath()).startswith('/robot/') or not p.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False or not p.IsA(UsdGeom.Mesh):continue
  m=UsdGeom.Mesh(p);v=np.asarray(m.GetPointsAttr().Get(),float);v=(np.c_[v,np.ones(len(v))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:2];v=(v-np.array(task['start_position_m'][:2]))@rot
  try:polys.append(Polygon(v[ConvexHull(v).vertices]))
  except QhullError:continue
 shape=unary_union(polys);(out/'recovery-footprint.wkb').write_bytes(shape.wkb)
 (out/'recovery-footprint-source.json').write_text(json.dumps({'source':'Actual enabled robot convex-part mesh XY projection union; no scene geometry','parts':len(polys),'area_m2':shape.area,'point_clearance_m':.005,'maximum_recovery_distance_m':.30},indent=2))

class StartupRecovery:
 def __init__(self,out):
  from shapely import wkb
  self.out=Path(out);self.shape=wkb.loads((self.out/'recovery-footprint.wkb').read_bytes()).buffer(.005);self.active=False;self.used=False;self.events=[];self.start=None;self.initial=None
 def record(self,event,**values):
  self.events.append(dict(event=event,**values));(self.out/'startup-recovery.json').write_text(json.dumps(self.events,indent=2))
 def raw_clear(self,points,p,yaw):
  if not len(points):return False
  d=points[:,:2]-p;c,s=np.cos(yaw),np.sin(yaw);local=d@np.array([[c,-s],[s,c]])
  return not np.any(contains_xy(self.shape,local[:,0],local[:,1]))
 def seen_sweep(self,layers,p,yaw):
  from g2_pose_planning import collision_mask
  # Unknown-space guard. Actual measured obstacle points are checked separately.
  rr,cc=self.planner.cell(p);offset=p-self.planner.xy(rr,cc);xx,yy=np.meshgrid(np.arange(-13,14)*.05-offset[0],np.arange(-13,14)*.05-offset[1]);c,s=np.cos(yaw),np.sin(yaw);x=c*xx+s*yy;y=-s*xx+c*yy;mask=contains_xy(self.shape,x,y)
  if rr<13 or cc<13 or rr+14>600 or cc+14>600:return False
  known=layers.observed|layers.robot_footprint
  return bool(np.all(known[rr-13:rr+14,cc-13:cc+14][mask]))
 def endpoint_safe(self,layers,p,yaw):
  from g2_pose_planning import collision_mask
  r,c=self.planner.cell(p);offset=p-self.planner.xy(r,c)
  if r<13 or c<13 or r+14>600 or c+14>600:return False
  free=(layers.observed|layers.robot_footprint)&~layers.local_occupied&~layers.global_occupied
  h=int(np.rint(yaw/(np.pi/8)))%16
  # Both continuous endpoint and the original quantized lattice start must pass.
  masks=[collision_mask(float(yaw),.05,.025,float(offset[0]),float(offset[1])),self.planner.footprint(h*np.pi/8)]
  return all(np.all(free[r-13:r+14,c-13:c+14][m]) for m in masks)
 def command(self,layers,p,yaw,points,planner,now):
  self.planner=planner;p=np.asarray(p);points=np.asarray(points).reshape(-1,3);points=points[(points[:,2]>=.035)&(points[:,2]<=1.8)]
  if self.initial is None:self.initial=p.copy()
  if not self.active:
   if self.used or planner.status!='start_pose_not_observed_free' or np.linalg.norm(p-self.initial)>.05:return None
   forward=np.array([np.cos(yaw),np.sin(yaw)]);candidate=None
   for distance in np.arange(.05,.301,.025):
    target=p+forward*distance
    if not all(self.endpoint_safe(layers,target+forward*offset,yaw) for offset in [-.015,0.,.015]):continue
    centers=[p+forward*t for t in np.arange(0,distance+.0025,.0025)]
    if all(self.raw_clear(points,center,yaw) and self.seen_sweep(layers,center,yaw) for center in centers):candidate=(target,float(distance));break
   if candidate is None:
    if not self.events:self.record('no_verified_straight_exit',time_s=now)
    return None
   self.used=True;self.active=True;self.start=p.copy();self.heading=yaw;self.target,distance=candidate;self.record('started',time_s=now,start_xy=p.tolist(),target_xy=self.target.tolist(),heading_rad=yaw,distance_m=distance,environment_input='current same-tick range returns and observed map only')
  forward=np.array([np.cos(self.heading),np.sin(self.heading)]);remaining=float(np.dot(self.target-p,forward));err=(self.heading-yaw+np.pi)%(2*np.pi)-np.pi
  if remaining<=.008 and self.endpoint_safe(layers,p,yaw):
   self.active=False;planner.route=[];planner.last=-100.;planner.status='startup_recovery_complete';self.record('finished',time_s=now,position_xy=p.tolist(),net_displacement_m=float(np.linalg.norm(p-self.start)));planner.motion_mode='startup_recovery_brake';return np.zeros(3)
  if np.linalg.norm(p-self.start)>.305:
   self.active=False;self.record('stopped_at_recovery_distance_limit',time_s=now,position_xy=p.tolist());planner.motion_mode='startup_recovery_limit_stop';return np.zeros(3)
  lookahead=min(max(remaining,.01),.03);centers=[p+forward*t for t in np.linspace(0,lookahead,13)]
  if abs(err)>.05 or not all(self.raw_clear(points,center,yaw) and self.seen_sweep(layers,center,yaw) for center in centers):
   self.active=False;self.record('stopped_by_live_guard',time_s=now,position_xy=p.tolist(),heading_error_rad=float(err));planner.motion_mode='startup_recovery_guard_stop';return np.zeros(3)
  planner.motion_mode='startup_recovery_forward';return np.array([min(.06,max(.025,remaining*1.2)),0.,np.clip(err,-.05,.05)])
