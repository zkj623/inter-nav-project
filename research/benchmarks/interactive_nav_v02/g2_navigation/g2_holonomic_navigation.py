"""Observed-only navigation for a tucked G2 with four steerable wheels.

Maintain the initial body heading; use lateral translation rather than rotating
an unmodelled swept body through furniture. No map prior or task waypoints.
"""
import heapq,math
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation,label,distance_transform_edt

class G2Navigator:
 def __init__(self,limits,resolution,heading):
  self.limits=limits;self.res=resolution;self.heading=heading;self.path=[];self.target=None;self.kind='wait';self.reason='initializing';self.last_plan=-1e9;self.visited={};self.last_motion=0.;self.motion_origin=None
  # Bounds measured from all tucked robot link collision vertices; add 2 cm.
  from g2_pose_planning import collision_mask
  self.footprint=collision_mask(float(heading),resolution)
 def cell(self,xy):return tuple(np.clip(np.floor((np.asarray(xy)-[self.limits[0][0],self.limits[1][0]])/self.res).astype(int)[::-1],0,599))
 def xy(self,rc):return np.array([self.limits[0][0]+(rc[1]+.5)*self.res,self.limits[1][0]+(rc[0]+.5)*self.res])
 def astar(self,s,g,safe,clearance):
  queue=[(0.,s)];cost={s:0.};parent={};closed=set()
  while queue:
   _,u=heapq.heappop(queue)
   if u in closed:continue
   if u==g:
    route=[u]
    while u in parent:u=parent[u];route.append(u)
    return [self.xy(r) for r in route[::-1]]
   closed.add(u)
   for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
    v=(u[0]+dr,u[1]+dc)
    if not (0<=v[0]<safe.shape[0] and 0<=v[1]<safe.shape[1]) or not safe[v]:continue
    if dr and dc and (not safe[u[0]+dr,u[1]] or not safe[u[0],u[1]+dc]):continue
    ng=cost[u]+math.hypot(dr,dc)*(1+.12/max(.03,clearance[v]))
    if ng<cost.get(v,1e30):cost[v]=ng;parent[v]=u;heapq.heappush(queue,(ng+math.hypot(v[0]-g[0],v[1]-g[1]),v))
  return []
 def update(self,layers,pos,goal,yaw,now,velocity):
  p=np.asarray(pos[:2]);goal=np.asarray(goal[:2]);distance=np.linalg.norm(p-goal)
  free=(layers.observed | layers.robot_footprint) & ~layers.local_occupied & ~layers.global_occupied
  safe=binary_erosion(free,structure=self.footprint,border_value=0)
  self.safe=safe;start=self.cell(p)
  if self.motion_origin is None or np.linalg.norm(p-self.motion_origin)>.06:self.motion_origin=p.copy();self.last_motion=now
  if distance<.48:self.path=[];self.kind='goal_braking';self.reason='task_goal_within_capture_radius';return np.zeros(3)
  if not safe[start]:
   self.kind='wait';self.reason='measured_body_footprint_not_observed_free';self.path=[];return np.zeros(3)
  need=now-self.last_plan>=1. or not self.path or any(not safe[self.cell(x)] for x in self.path)
  if self.target is not None and np.linalg.norm(p-self.target)<.18:
   if self.kind=='explore':self.visited[tuple(np.rint(self.target/.35).astype(int))]=int(layers.observed.sum())
   self.target=None;need=True
  if now-self.last_motion>4 and self.target is not None:
   self.visited[tuple(np.rint(self.target/.35).astype(int))]=int(layers.observed.sum())+80;self.target=None;need=True;self.last_motion=now
  if need:
   self.last_plan=now;components,_=label(safe,structure=np.ones((3,3)));reachable=components==components[start];clearance=distance_transform_edt(free)*self.res;g=self.cell(goal)
   yy,xx=np.indices(safe.shape);gx=self.limits[0][0]+(xx+.5)*self.res;gy=self.limits[1][0]+(yy+.5)*self.res;gd=np.hypot(gx-goal[0],gy-goal[1]);goal_cells=np.argwhere(reachable & safe & (gd<.475))
   if len(goal_cells):
    rr=min(goal_cells,key=lambda r:np.linalg.norm(self.xy(r)-p)+.3*gd[tuple(r)]);self.target=self.xy(rr);self.kind='navigate';self.reason='observed_full_body_route_to_goal_region' 
   else:
    # Reachable body centres at the edge of observed configuration space.
    boundary=reachable & ~binary_erosion(reachable,structure=np.ones((3,3)))
    rr=np.argwhere(boundary);choices=[];obs=int(layers.observed.sum());unknown_distance=distance_transform_edt(layers.observed)*self.res
    for r in rr[::max(1,len(rr)//180)]:
     xy=self.xy(r);key=tuple(np.rint(xy/.35).astype(int))
     if np.linalg.norm(xy-p)<.30 or obs-self.visited.get(key,-10000)<40:continue
     near_unknown=unknown_distance[tuple(r)]
     score=np.linalg.norm(xy-goal)+.35*np.linalg.norm(xy-p)+.5*near_unknown
     choices.append((score,xy))
    if choices:self.target=min(choices,key=lambda x:x[0])[1];self.kind='explore';self.reason='reachable_observed_body_frontier'
    else:self.target=None;self.kind='wait';self.reason='no_reachable_full_body_frontier'
   self.path=self.astar(start,self.cell(self.target),safe,clearance) if self.target is not None else []
  if not self.path:return np.zeros(3)
  route=np.asarray(self.path);i=int(np.linalg.norm(route-p,axis=1).argmin());self.path=self.path[max(0,i):];target=self.path[-1]
  # Short lookahead follows grid corners without cutting occupied cells.
  for point in self.path:
   if np.linalg.norm(point-p)>.12:target=point;break
  delta=target-p;norm=np.linalg.norm(delta)
  if norm<.015:return np.zeros(3)
  world=delta/norm*min(.14,norm*1.2,max(.035,np.linalg.norm(self.target-p)*.6))
  c,s=np.cos(yaw),np.sin(yaw);body=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1]])
  err=(self.heading-yaw+np.pi)%(2*np.pi)-np.pi
  if abs(err)>.12:return np.zeros(3)
  return np.r_[body,np.clip(1.5*err,-.12,.12)]
