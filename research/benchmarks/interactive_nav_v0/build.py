"""Build deterministic, geometry-screened multi-scene episodes and USD overlays."""
import argparse,json,math,heapq,hashlib,os
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from pxr import Usd,UsdGeom,UsdPhysics,Gf
R=.1
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def pathfind(mask,s,t):
 s=tuple(map(int,s));t=tuple(map(int,t))
 if not mask[s] or not mask[t]:return None
 q=[(0,0,s)];cost={s:0};prev={}
 while q:
  _,g,u=heapq.heappop(q)
  if g>cost[u]:continue
  if u==t:
   route=[u]
   while u!=s:u=prev[u];route.append(u)
   return route[::-1]
  for dy,dx in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
   v=(u[0]+dy,u[1]+dx)
   if not(0<=v[0]<mask.shape[0] and 0<=v[1]<mask.shape[1]) or not mask[v]:continue
   if dy and dx and not(mask[u[0]+dy,u[1]] and mask[u[0],u[1]+dx]):continue
   ng=g+math.hypot(dx,dy)
   if ng<cost.get(v,1e20):cost[v]=ng;prev[v]=u;heapq.heappush(q,(ng+math.dist(v,t),ng,v))
 return None
def length(route):return None if route is None else sum(math.dist(a,b)*R for a,b in zip(route,route[1:]))
def footprint(shape,xy,half):
 out=np.zeros(shape,np.uint8);cy,cx=xy;hy,hx=np.ceil(np.array(half[::-1])/R).astype(int)
 cv2.rectangle(out,(int(cx-hx),int(cy-hy)),(int(cx+hx),int(cy+hy)),1,-1);return out.astype(bool)
def mask_objects(free,items,skip=None):
 out=free.copy()
 for j,x in enumerate(items):
  if j!=skip:out[footprint(out.shape,x['cell'],np.array(x['dimensions_m'][:2])/2+.05)]=False
 return ndi.distance_transform_edt(out)*R>=.30
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 catalog=json.loads(a.catalog.read_text());ids=catalog['active_assets'];meta={x['asset']:x for x in catalog['objects']}
 selected=json.loads((a.data/'selected.json').read_text());rng=np.random.default_rng(20261118);summary=[]
 for si,row in enumerate(selected):
  dest=a.out/row['id'];dest.mkdir(exist_ok=True);d=np.load(a.data/(row['scene_id']+'-map.npz'));free=d['free'];clear=d['clearance'];origin=d['origin'];z=float(d['z']);nav=clear>=.30
  labels,n=ndi.label(nav);sizes=np.bincount(labels.ravel());sizes[0]=0;nav=labels==sizes.argmax();cells=np.argwhere(nav&(clear>.55));items=[];routes=[]
  if len(cells)<20:raise RuntimeError('insufficient navigable space '+row['id'])
  # Generate a diverse pool of routes; narrow sections of actual routes are candidate key positions.
  pool=[]
  for k in range(100):
   s,t=cells[rng.integers(len(cells),size=2)]
   if np.linalg.norm(s-t)*R<3:continue
   route=pathfind(nav,s,t)
   if route and len(route)>35:pool.append(route)
  for j in range(3):
   aid=ids[(si*3+j)%len(ids)];dim=np.ptp(np.load(a.objects/(aid+'.npz'))['vertices'],axis=0);candidates=[]
   for route in pool:
    for k in range(12,len(route)-12,3):
     cell=np.array(route[k]);c=float(clear[tuple(cell)])
     if any(np.linalg.norm(cell-x['cell'])*R<1.5 for x in items):continue
     fp=footprint(free.shape,cell,dim[:2]/2+.08)
     if not np.all(free[fp]):continue
     candidates.append((c+abs(k/len(route)-.5)*.1,cell,route,k))
   candidates.sort(key=lambda x:x[0]);chosen=None
   for score,cell,route,k in candidates[:500]:
    # Require a collision-free planar relocation witness; no claim of robotic executability.
    for angle in np.linspace(0,2*np.pi,16,endpoint=False):
     delta=np.array([np.sin(angle),np.cos(angle)])
     for distance in (.8,1.2,1.6):
      target=np.rint(cell+delta*distance/R).astype(int)
      if np.any(target<1) or np.any(target>=np.array(free.shape)-1):continue
      samples=[np.rint(cell+(target-cell)*t).astype(int) for t in np.linspace(0,1,25)]
      if not all(np.all(free[footprint(free.shape,c,dim[:2]/2+.08)]) for c in samples):continue
      if any(np.linalg.norm(target-x['cell'])*R<1 for x in items):continue
      chosen=(cell,route,k,target);break
     if chosen:break
    if chosen:break
   if chosen is None:raise RuntimeError(f'no placement {row["id"]} {aid}')
   cell,route,k,target=chosen;xy=origin+cell[::-1]*R
   items.append({'instance_id':f'O{j+1:02}','asset_id':aid,'name':meta[aid]['name'],'dimensions_m':dim.tolist(),'cell':cell.tolist(),'position_m':[float(xy[0]),float(xy[1]),z+.015],'yaw_deg':0,'mass_kg':1.,'static_friction':1.,'dynamic_friction':.9,'placement':'route bottleneck candidate','local_clearance_m':float(clear[tuple(cell)]),'relocation_witness_xy_m':(origin+target[::-1]*R).tolist(),'relocation_validation':'planar geometry only; robot contact execution pending'})
   routes.append(route)
  initial=mask_objects(free,items);tasks=[]
  for j,route in enumerate(routes):
   options=[]
   for candidate in [route]+pool:
    endpoints=[c for c in candidate if initial[tuple(c)] and all(np.linalg.norm(np.array(c)-x['cell'])*R>1 for x in items)]
    if len(endpoints)<2:continue
    s,t=endpoints[0],endpoints[-1];base=pathfind(nav,s,t)
    if base is None or length(base)<3:continue
    proximity=min(np.linalg.norm(np.array(c)-items[j]['cell'])*R for c in base)
    if proximity>.65:continue
    if any(np.allclose(origin+np.array(s)[::-1]*R,tt['start_position_m'][:2]) and np.allclose(origin+np.array(t)[::-1]*R,tt['goal_position_m'][:2]) for tt in tasks):continue
    options.append((proximity,s,t,base))
   if not options:raise RuntimeError('no >=3m task near object '+row['id']+' '+str(j))
   _,s,t,base=min(options,key=lambda x:x[0]);blocked=pathfind(initial,s,t)
   # Geometric reference when all introduced objects are absent.
   for reverse in (False,True):
    ss,tt=(t,s) if reverse else (s,t);route_xy=[(origin+np.array(c)[::-1]*R).tolist() for c in (base[::-1] if reverse else base)]
    start=route_xy[0];goal=route_xy[-1];yaw=math.atan2(route_xy[1][1]-start[1],route_xy[1][0]-start[0])
    tasks.append({'id':f'{row["id"]}-T{len(tasks)+1:02}','start_position_m':[*start,z+.55],'start_yaw_rad':yaw,'goal_position_m':[*goal,z+.55],'goal_tolerance_m':.5,'focus_object':items[j]['instance_id'],'initial_layout':'scene.usda','reference_path_m':length(base),'initial_detour_path_m':length(blocked),'geometric_class':'blocked_initially' if blocked is None else 'detour_available','reference_route_xy_m':route_xy,'validation':'geometry_screened; policy_rollout_pending'})
  if (dest/'scene.usda').exists():
   stage=Usd.Stage.Open(str(dest/'scene.usda'));stage.GetRootLayer().Clear()
  else:stage=Usd.Stage.CreateNew(str(dest/'scene.usda'))
  UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
  base=UsdGeom.Xform.Define(stage,'/World/Environment');base.GetPrim().GetReferences().AddReference(os.path.relpath(a.home/row['scene'],dest));base.AddScaleOp().Set(Gf.Vec3d(.01))
  # Remove source physics scenes so callers can use one explicitly configured physics world.
  for p in list(stage.Traverse()):
   if p.IsA(UsdPhysics.Scene):p.SetActive(False)
  for item in items:
   xf=UsdGeom.Xform.Define(stage,'/World/Objects/'+item['instance_id']);xf.GetPrim().GetReferences().AddReference(os.path.relpath(a.objects/(item['asset_id']+'.usdc'),dest));xf.ClearXformOpOrder();xf.AddTranslateOp(opSuffix='benchmark').Set(Gf.Vec3d(*item['position_m']));xf.GetPrim().SetCustomDataByKey('benchmark:asset_id',item['asset_id'])
  physics=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
  stage.GetRootLayer().Export(str(dest/'scene.usda'))
  record={**row,'objects':items,'tasks':tasks,'free_area_m2':round(float(free.sum()*R*R),2),'largest_robot_component_m2':round(float(nav.sum()*R*R),2),'grid_resolution_m':R,'robot_screening_radius_m':.3,'physics_validation':'pending','floor_area_definition':'union of near-horizontal ground projections within z +-0.18m; 0.1m raster; not architectural floor area'}
  dump(dest/'episodes.json',record);np.savez_compressed(dest/'evaluation_grid.npz',floor=d['floor'],blocked=d['blocked'],origin=origin,resolution=R)
  img=np.full((*free.shape,3),235,np.uint8);img[d['floor']==0]=255;img[d['blocked']!=0]=70
  for j,item in enumerate(items):
   fp=footprint(free.shape,item['cell'],np.array(item['dimensions_m'][:2])/2);img[fp]=(40,120,230)
   cv2.putText(img,item['instance_id'],tuple(np.array(item['cell'])[::-1]+3),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
  for task in tasks[::2]:
   pp=np.rint((np.array(task['reference_route_xy_m'])-origin)/R).astype(np.int32);cv2.polylines(img,[pp],False,(200,100,20),1);cv2.circle(img,tuple(pp[0]),3,(0,150,0),-1);cv2.circle(img,tuple(pp[-1]),3,(0,0,220),-1)
  cv2.imwrite(str(dest/'layout.png'),cv2.resize(np.flipud(img),None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
  summary.append({k:v for k,v in record.items() if k not in ('tasks','objects')});print(row['id'],row['floor_area_m2'],[x['asset_id'] for x in items],flush=True)
 dump(a.out/'scenes.json',summary);dump(a.out/'object_catalog.json',catalog)
 dump(a.out/'manifest.json',{'version':'0.1.0','seed':20261118,'scenes':len(summary),'tasks':len(summary)*6,'object_types':ids,'status':'geometry-screened prototype','physics_assumptions':'single rigid body, 1kg, friction 1.0/0.9 inherited from genie2-rsi; wheels/lids not articulated','files':{str(p.relative_to(a.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
if __name__=='__main__':main()
