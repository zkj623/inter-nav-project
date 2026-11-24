"""Original RSI Voronoi roadmap, realized by ordered full-body SE(2) search."""
import math,heapq,hashlib
from collections import OrderedDict
import numpy as np
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.mapping import MappingConfig,OccupancyGridMap,AStarMapPlanner,PlanningError
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiGraph
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import ConnectedVoronoiPathPlanner
from grutopia_extension.interactive_navigation.route_progress import trim_initial_attachment

class RSIVoronoiGuidance:
 def __init__(self,limits,res):
  self.limits=limits;self.res=res;self.cache=OrderedDict();self.records=[]
 def reference(self,center_free,known,start,goal):
  key=hashlib.blake2b(center_free.tobytes()+known.tobytes(),digest_size=16).digest()
  if key not in self.cache:
   cfg=MappingConfig(x_limits=tuple(self.limits[0]),y_limits=tuple(self.limits[1]),grid_resolution=self.res,robot_radius=self.res/10,obstacle_inflation_radius=self.res/10,static_obstacle_inflation_radius=self.res/10)
   occupancy=OccupancyGridMap(cfg);occupancy.static_occupied[:]=~center_free;occupancy.log_odds[:]=-4.;occupancy.observed[:]=known
   graph=SemanticVoronoiGraph(occupancy);graph.update();engine=ConnectedVoronoiPathPlanner(graph,waypoint_spacing=.30,observed_only=True,clearance_weight=.30);self.cache[key]=(engine,occupancy)
   if len(self.cache)>8:self.cache.popitem(last=False)
  engine,occupancy=self.cache[key];blocked=occupancy.inflated_mask()|~occupancy.observed;meta={'engine':'rsi_connected_clearance_weighted_voronoi','implementation':'grutopia_extension.interactive_navigation.go2_voronoi_navigation.ConnectedVoronoiPathPlanner','waypoint_spacing_m':.30,'clearance_weight_m':.30,'roadmap_occupancy':'online observed centers admitting at least one collision-safe Genie2 heading; no second robot-radius inflation'}
  try:
   points=engine.plan((*start,0.),(*goal,0.));meta['shared_component_attachment']=engine.shared_component_used
  except PlanningError as error:
   meta['engine']='rsi_grid_astar_fallback';meta['fallback_reason']=str(error)
   raw=AStarMapPlanner(occupancy,observed_only=True).plan((*start,0.),(*goal,0.));cells=[occupancy.world_to_cell(start)]+[occupancy.world_to_cell(point[:2]) for point in raw];cells=engine._sample_along_path(cells,blocked,self.res);points=tuple((*occupancy.cell_to_world(cell),0.) for cell in cells[1:])
   if not points or np.linalg.norm(np.array(points[-1][:2])-goal)>self.res:points=(*points,(*goal,0.))
  points,skipped=trim_initial_attachment(start,points,lambda a,b:AStarMapPlanner.segment_clear(occupancy.world_to_cell(a),occupancy.world_to_cell(b),blocked));meta['initial_attachment_points_skipped']=skipped;meta['skeleton_cells']=len(engine.graph.cached_snapshot().skeleton_cells);meta['reference_xy_m']=[list(point[:2]) for point in points]
  return np.array([point[:2] for point in points]),meta

def ordered_pose_route(planner,safe,swept,center_free,p,yaw,reference,free,limit=240000):
 """Visit each RSI reference waypoint in order; never replace it by a goal chord."""
 from scipy.ndimage import binary_erosion
 from g2_pose_planning import collision_mask
 res=planner.res;anchors=(np.asarray(reference)-np.array([planner.limits[0][0],planner.limits[1][0]]))/res-.5;anchors=anchors[:,::-1];n=len(anchors)
 if not n:return [],{'expanded_states':0,'reason':'empty_reference'}
 tolerance=.075/res;clearance=distance_transform_edt(center_free)*res;soft=1.+.30/np.maximum(clearance,res)
 suffix=np.zeros(n)
 for i in range(n-2,-1,-1):suffix[i]=suffix[i+1]+max(0.,np.linalg.norm(anchors[i+1]-anchors[i])-2*tolerance)
 def advance(r,c,k):
  while k<n and math.hypot(r-anchors[k,0],c-anchors[k,1])<=tolerance:k+=1
  return k
 def heuristic(r,c,k):return 0. if k>=n else max(0.,math.hypot(r-anchors[k,0],c-anchors[k,1])-tolerance)+suffix[k]
 r,c=planner.cell(p);initial_cell=(r,c);offset=p-planner.xy(r,c);patch=free[r-13:r+14,c-13:c+14]
 if patch.shape!=(27,27):return [],{'reason':'start_outside_checked_map'}
 actual_safe=binary_erosion(free,structure=planner.footprint(yaw));safe=np.concatenate([safe,actual_safe[None]],axis=0);headings=[h*np.pi/8 for h in range(16)]+[yaw];attachments={};actual_attachments={}
 for h in range(16):
  err=(headings[h]-yaw+np.pi)%(2*np.pi)-np.pi
  if abs(err)>np.pi/4+1e-9:continue
  angles=np.linspace(yaw,yaw+err,max(2,int(abs(err)/(np.pi/64))+2));mask=np.logical_or.reduce([collision_mask(float(a),res,.05) for a in angles]);attachments[h]=binary_erosion(free,structure=mask);actual_attachments[h]=all(np.all(patch[collision_mask(float(a),res,.05,float(offset[0]),float(offset[1]))]) for a in angles)
 k=advance(r,c,0);queue=[];cost={};parent={};closed=set();root_states=set()
 if actual_safe[r,c] and np.all(patch[collision_mask(float(yaw),res,.025,float(offset[0]),float(offset[1]))]):
  start=(r,c,16,k);cost[start]=0.;root_states.add(start);heapq.heappush(queue,(2.*heuristic(r,c,k),start))
 for h,valid in actual_attachments.items():
  if not valid or not safe[h,r,c]:continue
  start=(r,c,h,k);angle=abs((headings[h]-yaw+np.pi)%(2*np.pi)-np.pi);g=2.5*angle/(np.pi/8);cost[start]=g;root_states.add(start);heapq.heappush(queue,(g+2.*heuristic(r,c,k),start))
 if not queue:return [],{'reason':'measured_start_has_no_safe_pose_attachment'}
 while queue and len(closed)<limit:
  _,u=heapq.heappop(queue)
  if u in closed:continue
  r,c,h,k=u
  if k==n:
   states=[u]
   while u in parent:u=parent[u];states.append(u)
   route=[(planner.xy(r,c),headings[h]) for r,c,h,k in states[::-1]]
   first_heading=route[0][1];route.insert(0,(np.asarray(p).copy(),first_heading));route.insert(0,(np.asarray(p).copy(),float(yaw)))
   return route,{'expanded_states':len(closed),'reference_waypoints_visited_in_order':n,'waypoint_tolerance_m':.075,'translation_clearance_weight_m':.30,'weighted_cost_grid_units':cost[states[0]],'measured_heading_attachment':True,'local_search_heuristic_weight':2.}
  closed.add(u);neighbors=[]
  for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
   rr,cc=r+dr,c+dc
   if not (0<=rr<safe.shape[1] and 0<=cc<safe.shape[2]) or not safe[h,rr,cc]:continue
   if dr and dc and (not safe[h,r+dr,c] or not safe[h,r,c+dc]):continue
   angle=(math.atan2(dr,dc)-headings[h]+np.pi)%(2*np.pi)-np.pi;penalty=1.+4.*abs(math.sin(angle))+2.*max(0.,-math.cos(angle));weight=math.hypot(dr,dc)*penalty*.5*(soft[r,c]+soft[rr,cc]);neighbors.append(((rr,cc,h,advance(rr,cc,k)),weight))
  if h==16:
   for target,mask in attachments.items():
    valid=actual_attachments[target] if (r,c)==initial_cell else mask[r,c]
    if valid:neighbors.append(((r,c,target,k),2.5*abs((headings[target]-yaw+np.pi)%(2*np.pi)-np.pi)/(np.pi/8)))
  else:
   for direction,index in ((1,h),(-1,(h-1)%16)):
    valid=swept[index,r,c]
    if valid and (r,c)==initial_cell:
     a=headings[h];b=a+direction*np.pi/8;valid=all(np.all(patch[collision_mask(float(v),res,.05,float(offset[0]),float(offset[1]))]) for v in np.linspace(a,b,10))
    if valid and getattr(planner,'_observed_count',0)-planner.blocked_rotations.get((r,c,h,direction),-10000)>40:neighbors.append(((r,c,(h+direction)%16,k),2.5))
  for v,w in neighbors:
   g=cost[u]+w
   if g<cost.get(v,1e30):cost[v]=g;parent[v]=u;heapq.heappush(queue,(g+2.*heuristic(v[0],v[1],v[3]),v))
 return [],{'expanded_states':len(closed),'reason':'ordered_reference_has_no_full_body_pose_route' if not queue else 'pose_search_state_limit'}
