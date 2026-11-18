"""Fixed-layout G2 task suite: global connectivity cuts, parked-state witnesses, long routes."""
import argparse,json,math,itertools,heapq,os,hashlib
from pathlib import Path
import numpy as np,cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from pxr import Usd,UsdGeom,UsdPhysics,Gf
R=.025
STRUCT=np.ones((3,3),bool)
def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist()),encoding='utf-8')
def largest(mask):
 if not mask.any():return np.zeros_like(mask,dtype=bool)
 labels,n=ndi.label(mask,STRUCT);s=np.bincount(labels.ravel());s[0]=0;return labels==s.argmax()
def stamp(shape,kernel,cell):
 out=np.zeros(shape,bool);cy,cx=map(int,cell);hy,hx=np.array(kernel.shape)//2
 y0,y1=max(0,cy-hy),min(shape[0],cy+hy+1);x0,x1=max(0,cx-hx),min(shape[1],cx+hx+1)
 if y1>y0 and x1>x0:out[y0:y1,x0:x1]=kernel[y0-(cy-hy):y1-(cy-hy),x0-(cx-hx):x1-(cx-hx)]
 return out
def asset_kernels(path,yaw):
 m=np.load(path);v=m['vertices'].copy();angle=np.radians(yaw);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);v[:,:2]=v[:,:2]@rot.T
 n=int(np.ceil(np.abs(v[:,:2]).max()/R))+24;shape=(n*2+1,n*2+1);tri=v[m['faces']];kernels={}
 for name,top in [('full',1.8),('base',.4)]:
  mask=np.zeros(shape,np.uint8);keep=(tri[:,:,2].max(1)>.12)&(tri[:,:,2].min(1)<top)
  for p in np.rint(tri[keep,:,:2]/R+n).astype(np.int32):cv2.fillConvexPoly(mask,p,1)
  if not mask.any():raise ValueError('No base-height geometry '+str(path))
  kernels[name]=mask.astype(bool)
 kernels['fit']=ndi.distance_transform_edt(~kernels['full'])*R<=.05
 # Extra optimistic radius and 8-connectivity intentionally admit narrow/diagonal bypasses.
 kernels['optimistic']=ndi.distance_transform_edt(~kernels['base'])*R<=.25
 kernels['nominal']=ndi.distance_transform_edt(~kernels['full'])*R<=.36
 return kernels
def nav_with(nav,items,kernels,kind,parked=()):
 result=nav.copy()
 for j,o in enumerate(items):result[stamp(nav.shape,kernels[(o['asset_id'],o['yaw_deg'])][kind],o['parking_cell'] if j in parked else o['cell'])]=False
 return result
def graph(mask):
 cells=np.argwhere(mask);index=np.full(mask.shape,-1,np.int32);index[mask]=np.arange(len(cells));rows=[];cols=[];weights=[]
 for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
  yy=cells[:,0]+dy;xx=cells[:,1]+dx;valid=(yy>=0)&(yy<mask.shape[0])&(xx>=0)&(xx<mask.shape[1]);ids=np.flatnonzero(valid);v=index[yy[ids],xx[ids]];ids=ids[v>=0];v=v[v>=0]
  if dy and dx:
   ok=mask[cells[ids,0]+dy,cells[ids,1]]&mask[cells[ids,0],cells[ids,1]+dx];ids=ids[ok];v=v[ok]
  rows.extend([ids,v]);cols.extend([v,ids]);weights.extend([np.full(len(ids),math.hypot(dx,dy)*R)]*2)
 return coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(cells),len(cells))).tocsr(),cells,index
def route(cells,pred,s,t):
 ids=[int(t)]
 while ids[-1]!=s:
  j=int(pred[ids[-1]])
  if j<0:raise ValueError('No route')
  ids.append(j)
 return cells[ids[::-1]]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--fine',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only');ap.add_argument('--reuse-layouts',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 catalog=json.loads((a.old/'object_catalog.json').read_text());meta={x['asset']:x for x in catalog['objects']};kernels={};rng=np.random.default_rng(20261119)
 for aid in catalog['active_assets']:
  for yaw in (0,90):kernels[aid,yaw]=asset_kernels(a.objects/(aid+'.npz'),yaw)
 for oldfile in sorted(a.old.glob('S*/episodes.json')):
  old=json.loads(oldfile.read_text());sid=old['id']
  if a.only and sid!=a.only:continue
  out=a.out/sid;out.mkdir(exist_ok=True);data=np.load(a.fine/(sid+'.npz'));floor=data['floor'].astype(bool);origin=data['origin'];bands=data['obstacle_bands'].astype(bool)
  static=floor&~bands.any(0);optimistic=floor&~bands[0]
  nominal=largest(ndi.distance_transform_edt(static)*R>=.36)
  optimistic=ndi.distance_transform_edt(optimistic)*R>=.25
  points=np.argwhere(nominal);clear=ndi.distance_transform_edt(static)*R;items=[]
  # Every original object type remains in the scene. Add a rigid chair only if needed for a genuine cut.
  aids=[x['asset_id'] for x in old['objects']]
  candidates=points[(points[:,0]%4==0)&(points[:,1]%4==0)&(clear[nominal]<.95)]
  if len(candidates)>900:candidates=candidates[rng.choice(len(candidates),900,replace=False)]
  def evaluate(aid,existing,require_cut):
   choices=[];nav=nav_with(optimistic,existing,kernels,'optimistic');previous,_=ndi.label(nav,STRUCT)
   for yaw in (0,90):
    kk=kernels[aid,yaw];fit=cv2.filter2D((~static).astype(np.float32),-1,kk['fit'].astype(np.float32),borderType=cv2.BORDER_CONSTANT)<.5
    for cell in candidates[fit[candidates[:,0],candidates[:,1]]]:
     if any(np.linalg.norm(cell-np.array(x['cell']))*R<1.2 for x in existing):continue
     blocked=nav.copy();blocked[stamp(nav.shape,kk['optimistic'],cell)]=False;labels,_=ndi.label(blocked,STRUCT)
     counts=np.bincount(labels[nominal]);counts[0]=0;major=np.sort(counts[counts>int(.35/R**2)])
     score=int(major[-2]) if len(major)>1 else 0
     choices.append((score,float(clear[tuple(cell)]),yaw,cell.copy()))
   if not choices:return None
   best=max(choices,key=lambda x:(x[0],-x[1]))
   if require_cut and best[0]==0:return None
   return best
  if a.reuse_layouts:items=json.loads((out/'episodes.json').read_text())['objects'];aids=[]
  for aid in aids:
   choice=evaluate(aid,items,False)
   if choice is None:raise RuntimeError(sid+' cannot fit '+aid)
   score,c,yaw,cell=choice;items.append({'asset_id':aid,'name':meta[aid]['name'],'cell':cell.tolist(),'yaw_deg':yaw,'cut_score':score})
  initial=nav_with(optimistic,items,kernels,'optimistic');labels,_=ndi.label(initial,STRUCT);sizes=np.bincount(labels[nominal]);sizes[0]=0
  if sum(sizes>int(.6/R**2))<2:
   for aid in ('chair_dining','chair_wheeled'):
    choice=evaluate(aid,items,True)
    if choice:
     score,c,yaw,cell=choice;items.append({'asset_id':aid,'name':meta[aid]['name'],'cell':cell.tolist(),'yaw_deg':yaw,'cut_score':score});break
   else:raise RuntimeError(sid+' no genuine connectivity cut found')
  # Park objects at collision-free perimeter points, outside the central route corridor where possible.
  parking_points=np.argwhere(static&(clear>.12));parking_candidates=parking_points[(parking_points[:,0]%4==0)&(parking_points[:,1]%4==0)]
  for j,item in enumerate([] if a.reuse_layouts else items):
   kk=kernels[item['asset_id'],item['yaw_deg']];fit=cv2.filter2D((~static).astype(np.float32),-1,kk['fit'].astype(np.float32),borderType=cv2.BORDER_CONSTANT)<.5
   choices=[]
   for c in parking_candidates[fit[parking_candidates[:,0],parking_candidates[:,1]]]:
    distance=np.linalg.norm(c-np.array(item['cell']))*R
    if distance<.9:continue
    if any(np.linalg.norm(c-np.array(x['cell']))*R<1 for x in items):continue
    if any(np.linalg.norm(c-np.array(x['parking_cell']))*R<1 for x in items[:j]):continue
    # Preserve the navigation network: parking in the widest corridor can itself seal it.
    proposed=nominal.copy()
    for previous in items[:j]:proposed[stamp(nominal.shape,kernels[previous['asset_id'],previous['yaw_deg']]['nominal'],previous['parking_cell'])]=False
    proposed[stamp(nominal.shape,kk['nominal'],c)]=False
    connected=largest(proposed)
    choices.append((float(connected.sum())*R*R-.02*distance,c))
   if not choices:raise RuntimeError(sid+' no parking spot')
   item['parking_cell']=max(choices,key=lambda x:x[0])[1].tolist()
  final=nav_with(nominal,items,kernels,'nominal',range(len(items)));final=largest(final)
  initial_nom=nav_with(nominal,items,kernels,'nominal');initial_opt=nav_with(optimistic,items,kernels,'optimistic');initial_labels,_=ndi.label(initial_opt,STRUCT)
  # Sample farthest-point landmarks on the cleared scene. Distances are actual grid geodesics.
  g,cells,index=graph(final);safe=initial_nom&final&(clear>.48);eligible=np.flatnonzero(safe[tuple(cells.T)])
  if len(eligible)<20:raise RuntimeError(sid+' insufficient endpoints')
  landmark_ids=[int(eligible[0])];minimum=np.full(len(cells),np.inf)
  for k in range(32):
   dist=dijkstra(g,indices=landmark_ids[-1]);minimum=np.minimum(minimum,dist);remaining=minimum[eligible];nxt=int(eligible[np.argmax(remaining)])
   if minimum[nxt]<.7:break
   landmark_ids.append(nxt)
  distances,preds=dijkstra(g,indices=landmark_ids,return_predecessors=True)
  pairs=[];pair_meta={};subset_labels=[]
  for count in range(1,len(items)+1):
   for subset in itertools.combinations(range(len(items)),count):
    ll,_=ndi.label(nav_with(optimistic,items,kernels,'optimistic',subset),STRUCT);subset_labels.append((count,list(subset),ll))
  for ii,s in enumerate(landmark_ids):
   for jj,t in enumerate(landmark_ids):
    if ii==jj:continue
    l=float(distances[ii,t]);sc=tuple(cells[s]);tc=tuple(cells[t]);ls=int(initial_labels[sc]);lt=int(initial_labels[tc])
    if ls and lt and ls!=lt and np.isfinite(l) and l>=4:
     solutions=[(n,subset) for n,subset,ll in subset_labels if ll[sc] and ll[sc]==ll[tc]]
     if not solutions:continue
     minimum=solutions[0][0];solutions=[ss for n,ss in solutions if n==minimum];pairs.append((l,ii,s,t));pair_meta[s,t]={'minimum':minimum,'solutions':solutions,'components':tuple(sorted((ls,lt)))}
  pairs.sort(reverse=True);chosen=[];covered_objects=set();covered_components=set()
  while pairs and len(chosen)<8:
   if not chosen:candidate=pairs[0]
   else:
    def diversity_key(c):
     m=pair_meta[c[2],c[3]];novel=max(len(set(ss)-covered_objects) for ss in m['solutions']);return (novel,int(m['components'] not in covered_components),c[0])
    candidate=max(pairs,key=diversity_key)
   l,ii,s,t=candidate;chosen.append(candidate);m=pair_meta[s,t];witness=max(m['solutions'],key=lambda ss:len(set(ss)-covered_objects));m['witness']=witness;covered_objects.update(witness);covered_components.add(m['components'])
   pairs=[c for c in pairs if not ((np.linalg.norm(cells[s]-cells[c[2]])*R<.85 and np.linalg.norm(cells[t]-cells[c[3]])*R<.85) or (np.linalg.norm(cells[s]-cells[c[3]])*R<.85 and np.linalg.norm(cells[t]-cells[c[2]])*R<.85))]
  if len(chosen)<8:raise RuntimeError(sid+' fewer than 8 mandatory tasks: '+str(len(chosen)))
  tasks=[]
  for l,ii,s,t in chosen:
   sc=tuple(cells[s]);tc=tuple(cells[t]);minimum_moved=pair_meta[s,t]['minimum'];witness=pair_meta[s,t]['witness']
   rr=route(cells,preds[ii],s,t);tasks.append({'class':'interaction_required','initial_optimistic_connected':False,'minimum_relocations_in_optimistic_model':minimum_moved,'optimistic_relocation_subset':witness,'reference_route_cells':rr.tolist(),'cleared_route_length_m':l,'start_cell':list(sc),'goal_cell':list(tc)})
  # Controls: a shortest path in the empty-of-added-objects scene remains collision-free initially.
  basegraph,basecells,baseindex=graph(nominal);control_candidates=np.argwhere(initial_nom&final&(clear>.48))[::max(1,int(safe.sum()/45))];controls=[]
  for sc in control_candidates:
   s=int(baseindex[tuple(sc)]);dd,pp=dijkstra(basegraph,indices=s,return_predecessors=True)
   for tc in control_candidates:
    t=int(baseindex[tuple(tc)]);l=float(dd[t])
    if not np.isfinite(l) or l<2:continue
    rr=route(basecells,pp,s,t)
    if initial_nom[tuple(rr.T)].all():controls.append((l,sc.copy(),tc.copy(),rr))
  controls.sort(key=lambda x:-x[0]);selected_controls=[]
  for c in controls:
   if any((np.linalg.norm(c[1]-d[1])*R<1 and np.linalg.norm(c[2]-d[2])*R<1) or (np.linalg.norm(c[1]-d[2])*R<1 and np.linalg.norm(c[2]-d[1])*R<1) for d in selected_controls):continue
   selected_controls.append(c)
   if len(selected_controls)==2:break
  if len(selected_controls)<2:raise RuntimeError(sid+' fewer than 2 optimal noninteraction controls')
  for l,sc,tc,rr in selected_controls:tasks.append({'class':'no_interaction_optimal','initial_optimistic_connected':True,'minimum_relocations_in_optimistic_model':0,'reference_route_cells':rr.tolist(),'cleared_route_length_m':l,'start_cell':sc.tolist(),'goal_cell':tc.tolist(),'optimality':'shortest route with added objects absent is feasible with all added objects in initial poses'})
  for j,item in enumerate(items):
   v=np.load(a.objects/(item['asset_id']+'.npz'))['vertices'];item.update(instance_id=f'O{j+1:02}',dimensions_m=np.ptp(v,axis=0).tolist(),position_m=[*(origin+np.array(item['cell'])[::-1]*R).tolist(),float(data['z'])+.015],parking_position_m=[*(origin+np.array(item['parking_cell'])[::-1]*R).tolist(),float(data['z'])+.015],mass_kg=1.,static_friction=1.,dynamic_friction=.9)
  for j,t in enumerate(tasks):
   xy=origin+np.array(t.pop('reference_route_cells'))[:,::-1]*R;t.update(id=f'{sid}-T{j+1:02}',start_position_m=[*xy[0].tolist(),0.0],goal_position_m=[*xy[-1].tolist(),0.0],start_yaw_rad=float(math.atan2(*(xy[1]-xy[0])[::-1])),reference_route_xy_m=xy.tolist(),goal_tolerance_m=.5,validation='G2 planar navigation model; robot manipulation rollout pending')
  record={k:v for k,v in old.items() if k not in ('objects','tasks','physics_validation','physics_validation_report','robot_screening_radius_m','largest_robot_component_m2')};record.update(version='0.2.0',objects=items,tasks=tasks,grid_resolution_m=R,robot='Genie2/G2 CRS',nominal_radius_m=.36,optimistic_radius_m=.25,largest_robot_component_m2=float(nominal.sum()*R*R),task_count=10,interaction_required=8,no_interaction_optimal=2,physics_validation='pending',cleared_state='all objects moved to recorded parking poses; not removed',long_route_sampling='farthest-point landmarks, descending geodesic length',robot_rollout_validated=False)
  dump(out/'episodes.json',record);np.savez_compressed(out/'evaluation_grid.npz',floor=floor,blocked=~static,base_blocked=bands[0],origin=origin,resolution=R,nominal=nominal,optimistic=optimistic,initial_nominal=initial_nom,initial_optimistic=initial_opt,cleared_nominal=final)
  for parked,filename in [(False,'scene.usda'),(True,'cleared_scene.usda')]:
   dest=out/filename
   if dest.exists():stage=Usd.Stage.Open(str(dest));stage.GetRootLayer().Clear()
   else:stage=Usd.Stage.CreateNew(str(dest))
   UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim());env=UsdGeom.Xform.Define(stage,'/World/Environment');env.GetPrim().GetReferences().AddReference(os.path.relpath(a.home/old['scene'],out));env.AddScaleOp().Set(Gf.Vec3d(.01))
   for p in list(stage.Traverse()):
    if p.IsA(UsdPhysics.Scene):p.SetActive(False)
   for item in items:
    xf=UsdGeom.Xform.Define(stage,'/World/Objects/'+item['instance_id']);xf.GetPrim().GetReferences().AddReference(os.path.relpath(a.objects/(item['asset_id']+'.usdc'),out));xf.ClearXformOpOrder();xf.AddTranslateOp(opSuffix='benchmark').Set(Gf.Vec3d(*item['parking_position_m' if parked else 'position_m']));xf.AddRotateZOp(opSuffix='benchmark').Set(float(item['yaw_deg']))
   phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81);stage.GetRootLayer().Save()
  print(sid,'objects',len(items),'required',8,'control',2,'mandatory_m',round(min(t['cleared_route_length_m'] for t in tasks[:8]),2),round(max(t['cleared_route_length_m'] for t in tasks[:8]),2),flush=True)
if __name__=='__main__':main()
