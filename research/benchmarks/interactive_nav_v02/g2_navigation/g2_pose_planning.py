"""Forward-preferred SE(2) navigation with explicit lateral escape actions."""
import heapq,math,json,os
from pathlib import Path
from scipy.spatial import ConvexHull
from functools import lru_cache
import numpy as np
from scipy.ndimage import binary_erosion,uniform_filter

_VERTICES=np.asarray(json.loads(Path(os.environ.get('G2_FOOTPRINT_PATH',str(Path(__file__).with_name('g2_collision_footprint.json')))).read_text())['vertices'])
# Wheel steering sweeps remain within these per-wheel 7 cm discs.
_DISCS=np.array([[x+.07*np.cos(a),y+.07*np.sin(a)] for x in (-.23,.23) for y in (-.218,.218) for a in np.linspace(0,2*np.pi,32)])
_EQ=ConvexHull(np.vstack((_VERTICES,_DISCS))).equations
@lru_cache(maxsize=128)
def collision_mask(a,res=.05,margin=.025,dx=0.,dy=0.):
 xx,yy=np.meshgrid(np.arange(-13,14)*res-dx,np.arange(-13,14)*res-dy);c,s=np.cos(a),np.sin(a);x=c*xx+s*yy;y=-s*xx+c*yy
 return np.all(x[:,:,None]*_EQ[:,0]+y[:,:,None]*_EQ[:,1]+_EQ[:,2]<=margin,axis=2)

class PoseGoalPlanner:
 def __init__(self,limits,res=.05):self.limits=limits;self.res=res;self.route=[];self.last=-100.;self.status='unplanned';self.checked=-100.;self.visited={};self.route_kind='explore';self.motion_mode='wait';self.blocked_rotations={}
 def cell(self,p):return tuple(np.clip(np.floor((np.asarray(p)-[self.limits[0][0],self.limits[1][0]])/self.res).astype(int)[::-1],0,599))
 def xy(self,r,c):return np.array([self.limits[0][0]+(c+.5)*self.res,self.limits[1][0]+(r+.5)*self.res])
 def footprint(self,a):return collision_mask(float(a),self.res)
 def plan(self,layers,p,yaw,goal,now):
  free=(layers.observed | layers.robot_footprint)&~layers.local_occupied&~layers.global_occupied
  safe=[binary_erosion(free,structure=self.footprint(h*np.pi/8)) for h in range(16)]
  swept=[binary_erosion(free,structure=collision_mask(h*np.pi/8,self.res,.05)|collision_mask((h+.5)*np.pi/8,self.res,.05)|collision_mask((h+1)*np.pi/8,self.res,.05)) for h in range(16)]
  r,c=self.cell(p);h=int(np.rint(yaw/(np.pi/8)))%16;start=(r,c,h)
  if not safe[h][r,c]:self.status='start_pose_not_observed_free';return []
  yy,xx=np.indices(free.shape);gd=np.hypot(self.limits[0][0]+(xx+.5)*self.res-goal[0],self.limits[1][0]+(yy+.5)*self.res-goal[1]);gm=gd<.475
  info=uniform_filter((~layers.observed).astype(float),size=31,mode='constant')*2.4
  best_frontier=None;best_score=1e30;observed_count=int(layers.observed.sum())
  queue=[(0.,start)];cost={start:0.};parent={};closed=set()
  def unpack(u):
   path=[u]
   while u in parent:u=parent[u];path.append(u)
   return [(self.xy(r,c),h*np.pi/8) for r,c,h in path[::-1]]
  while queue and len(closed)<180000:
   _,u=heapq.heappop(queue)
   if u in closed:continue
   r,c,h=u
   if gm[r,c]:
    self.status='observed_pose_route';self.route_kind='navigate';return unpack(u)
   xy=self.xy(r,c);key=tuple(np.rint(xy/.35).astype(int))
   if np.linalg.norm(xy-p)>.35 and observed_count-self.visited.get(key,-10000)>40:
    score=1.25*gd[r,c]+.25*cost[u]*self.res-.8*info[r,c]
    if score<best_score:best_score=score;best_frontier=u
   closed.add(u);neighbors=[]
   for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
    rr,cc=r+dr,c+dc
    if not (0<=rr<600 and 0<=cc<600) or not safe[h][rr,cc]:continue
    if dr and dc and (not safe[h][r+dr,c] or not safe[h][r,c+dc]):continue
    # Forward translation has unit cost. Sideways/backward actions remain
    # available when the observed swept body cannot turn in the available room.
    angle=(math.atan2(dr,dc)-h*np.pi/8+np.pi)%(2*np.pi)-np.pi
    forward=math.cos(angle);side=abs(math.sin(angle))
    penalty=1.+4.*side+2.*max(0.,-forward)
    neighbors.append(((rr,cc,h),math.hypot(dr,dc)*penalty))
   for direction,index in ((1,h),(-1,(h-1)%16)):
    key=(r,c,h,direction)
    if swept[index][r,c] and observed_count-self.blocked_rotations.get(key,-10000)>40:
     neighbors.append(((r,c,(h+direction)%16),2.5))
   for v,w in neighbors:
    g=cost[u]+w
    if g<cost.get(v,1e30):cost[v]=g;parent[v]=u;heapq.heappush(queue,(g+max(0.,gd[v[0],v[1]]-.475)/self.res,v))
  if best_frontier is not None:
   self.status='observed_pose_frontier';self.route_kind='explore';return unpack(best_frontier)
  self.status='no_observed_pose_route';return []
 def command(self,layers,p,yaw,goal,now):
  if np.linalg.norm(p-goal)<.48:self.motion_mode='goal_braking';return np.zeros(3)
  if self.route and now-self.checked>.3:
   self.checked=now;free=(layers.observed|layers.robot_footprint)&~layers.local_occupied&~layers.global_occupied
   for xy,a in self.route[:20]:
    r,c=self.cell(xy);f=self.footprint(a)
    if r<13 or c<13 or r+14>600 or c+14>600 or not np.all(free[r-13:r+14,c-13:c+14][f]):
     self.route=[];self.last=-100.;break
  if self.route and self.route_kind=='explore' and np.linalg.norm(self.route[-1][0]-p)<.15 and abs((self.route[-1][1]-yaw+np.pi)%(2*np.pi)-np.pi)<.05:
   self.visited[tuple(np.rint(self.route[-1][0]/.35).astype(int))]=int(layers.observed.sum());self.route=[];self.last=-100.
  if not self.route and now-self.last>2:
   self.last=now;self.route=self.plan(layers,p,yaw,goal,now)
  if not self.route:self.motion_mode='wait';return None
  wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
  while len(self.route)>1 and np.linalg.norm(self.route[0][0]-p)<.025 and abs(wrap(self.route[0][1]-yaw))<=.05:self.route.pop(0)
  xy,a=self.route[0];err=wrap(a-yaw);delta=xy-p
  # Rotate at the planned corner rather than translating while turning.
  if abs(err)>.05:
   c,s=np.cos(yaw),np.sin(yaw)
   def align_center():
    self.motion_mode='align_rotation_center';hold=delta/max(np.linalg.norm(delta),1e-9)*max(.045,min(.06,np.linalg.norm(delta)*1.2))
    return np.array([c*hold[0]+s*hold[1],-s*hold[0]+c*hold[1],0.])
   if np.linalg.norm(delta)>.04:return align_center()
   free=(layers.observed|layers.robot_footprint)&~layers.local_occupied&~layers.global_occupied
   r,col=self.cell(p)
   angles=np.linspace(yaw,yaw+err,max(2,int(abs(err)/(np.pi/64))+2))
   valid=r>=13 and col>=13 and r+14<=600 and col+14<=600
   if valid:
    patch=free[r-13:r+14,col-13:col+14]
    offset=p-self.xy(r,col)
    valid=all(np.all(patch[collision_mask(float(v),self.res,.05,float(offset[0]),float(offset[1]))]) for v in angles)
   if not valid:
    if np.linalg.norm(delta)>.012:return align_center()
    heading=int(np.rint(yaw/(np.pi/8)))%16
    self.blocked_rotations[(r,col,heading,1 if err>0 else -1)]=int(layers.observed.sum())
    self.route=[];self.last=-100.;self.status='rotation_sweep_not_observed_free';self.motion_mode='wait';return np.zeros(3)
   self.motion_mode='turn_in_place';omega=np.sign(err)*max(.075,min(.22,1.1*abs(err)))
   return np.array([0.,0.,omega])
  # Extend the target only through a straight, constant-heading lattice segment.
  # This avoids slowing at every 5 cm cell without cutting grid corners.
  if len(self.route)>1:
   direction=delta/max(np.linalg.norm(delta),1e-9)
   if np.linalg.norm(delta)<.012:
    direction=(self.route[1][0]-p);direction/=max(np.linalg.norm(direction),1e-9)
   for candidate,heading in self.route[1:]:
    d=candidate-p;length=np.linalg.norm(d)
    if abs(wrap(heading-a))>.01 or length>.20 or np.dot(d,direction)<0 or abs(direction[0]*d[1]-direction[1]*d[0])>.008:break
    xy=candidate;delta=d
  norm=np.linalg.norm(delta)
  if norm<.012:return np.zeros(3)
  v=delta/norm*min(.14,norm*1.5);c,s=np.cos(yaw),np.sin(yaw)
  body=np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])
  self.motion_mode='forward' if body[0]>0 and abs(body[1])<.45*body[0] else 'lateral_escape' if abs(body[1])>=abs(body[0]) else 'reverse_escape'
  return np.r_[body,np.clip(err,-.08,.08)]
