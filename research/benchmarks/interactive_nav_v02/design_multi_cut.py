"""Design one layout per scene with several independently meaningful connectivity cuts."""
import argparse,json,itertools,math,shutil
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from scipy import ndimage as ndi
from upgrade_connectivity import asset_kernels,stamp,largest,nav_with,dump,R,STRUCT
from diversify_tasks import sample

def meaningful(labels,core,minimum=.65):
 counts=np.bincount(labels[core]);counts[0]=0
 ids=np.flatnonzero(counts*R*R>=minimum)
 return ids,counts

def design(old,fine,kernels,asset_names,objects):
 sid=old['id'];floor=fine['floor'].astype(bool);bands=fine['obstacle_bands'].astype(bool);origin=fine['origin'];static=floor&~bands.any(0);clear=ndi.distance_transform_edt(static)*R
 nominal=largest(clear>=.36);optimistic=ndi.distance_transform_edt(floor&~bands[0])*R>=.25;core=nominal&(clear>.40)
 points=np.argwhere(nominal);candidates=points[(points[:,0]%3==0)&(points[:,1]%3==0)&(clear[nominal]<1.2)]
 rng=np.random.default_rng(20261120+int(sid[1:]));
 if len(candidates)>1500:candidates=candidates[rng.choice(len(candidates),1500,replace=False)]
 aids=[o['asset_id'] for o in old['objects']][:3]
 if int(sid[1:])>5:aids=aids[1:]+aids[:1]
 items=[];fit_cache={}
 def fit(key):
  if key not in fit_cache:fit_cache[key]=cv2.filter2D((~static).astype(np.float32),-1,kernels[key]['fit'].astype(np.float32),borderType=cv2.BORDER_CONSTANT)<.5
  return fit_cache[key]
 def choose(aid):
  current=nav_with(optimistic,items,kernels,'optimistic');before,_=ndi.label(current,STRUCT);before_ids,before_counts=meaningful(before,core)
  occupied=np.zeros_like(static)
  for obj in items:occupied|=stamp(static.shape,kernels[obj['asset_id'],obj['yaw_deg']]['fit'],obj['cell'])
  best=None
  for yaw in (0,90):
   kk=kernels[aid,yaw]
   for cell in candidates[fit((aid,yaw))[tuple(candidates.T)]]:
    if (stamp(static.shape,kk['fit'],cell)&occupied).any():continue
    mask=current.copy();mask[stamp(static.shape,kk['optimistic'],cell)]=False;labels,_=ndi.label(mask,STRUCT);ids,counts=meaningful(labels,core)
    # Evaluate new splits, not an old split left unchanged by this candidate.
    retained=core&mask;old_sizes=np.bincount(before[retained]);old_sizes[0]=0;new_sizes=np.bincount(labels[retained]);new_sizes[0]=0
    gain=float((np.square(old_sizes.astype(float)).sum()-np.square(new_sizes.astype(float)).sum())*R**4)
    score=(len(ids),gain,float(np.sort(counts[ids])[0])*R*R if len(ids) else 0,-float(clear[tuple(cell)]))
    if best is None or score>best[0]:best=(score,yaw,cell.tolist(),len(before_ids))
  return best
 for aid in aids:
  choice=choose(aid)
  if choice is None:raise RuntimeError(sid+' cannot place '+aid)
  score,yaw,cell,before_n=choice;items.append({'asset_id':aid,'name':asset_names[aid],'yaw_deg':yaw,'cell':cell,'regions_before_placement':before_n,'regions_after_placement':score[0],'incremental_split_score_m4':score[1]})
  print(sid,aid,'regions',before_n,'->',score[0],flush=True)
 # If three types cannot create enough cuts, permit another movable catalog object.
 for _ in range(2):
  initial=nav_with(optimistic,items,kernels,'optimistic');labels,_=ndi.label(initial,STRUCT);region_ids,counts=meaningful(labels,core)
  if len(region_ids)>=3:break
  choices=[]
  for aid in ('chair_dining','box_reinforced','stool_rectangular','bin_lidded'):
   c=choose(aid)
   if c:choices.append((c[0],aid,c))
  if not choices:break
  _,aid,choice=max(choices,key=lambda x:x[0]);score,yaw,cell,before_n=choice
  if score[0]<=len(region_ids):break
  items.append({'asset_id':aid,'name':asset_names[aid],'yaw_deg':yaw,'cell':cell,'regions_before_placement':before_n,'regions_after_placement':score[0],'incremental_split_score_m4':score[1]})
  print(sid,'added',aid,'regions',before_n,'->',score[0],flush=True)
 old_ids={o['asset_id']:o['instance_id'] for o in old['objects'][:3]};used_ids=set();extra=4
 for obj in items:
  preferred=old_ids.get(obj['asset_id'])
  if preferred and preferred not in used_ids:obj['instance_id']=preferred
  else:obj['instance_id']=f'O{extra:02}';extra+=1
  used_ids.add(obj['instance_id'])
 items.sort(key=lambda o:o['instance_id'])
 initial_opt=nav_with(optimistic,items,kernels,'optimistic');initial_labels,_=ndi.label(initial_opt,STRUCT);region_ids,counts=meaningful(initial_labels,core)
 if len(region_ids)<3:raise RuntimeError(sid+f' only {len(region_ids)} meaningful initial regions')
 initial_nom=nav_with(nominal,items,kernels,'nominal');safe_initial=initial_nom&(clear>.40)
 # Park near the scene perimeter; preserve endpoint space in every initial region.
 parking=np.argwhere(static&(clear>.12));parking=parking[(parking[:,0]%4==0)&(parking[:,1]%4==0)]
 for j,obj in enumerate(items):
  kk=kernels[obj['asset_id'],obj['yaw_deg']];choices=[]
  for cell in parking[fit((obj['asset_id'],obj['yaw_deg']))[tuple(parking.T)]]:
   if any(np.linalg.norm(cell-np.array(o['cell']))*R<.9 for o in items):continue
   if any(np.linalg.norm(cell-np.array(o['parking_cell']))*R<.9 for o in items[:j]):continue
   proposed=nominal.copy()
   for o in items[:j]:proposed[stamp(static.shape,kernels[o['asset_id'],o['yaw_deg']]['nominal'],o['parking_cell'])]=False
   proposed[stamp(static.shape,kk['nominal'],cell)]=False;connected=largest(proposed);sizes=np.bincount(initial_labels[connected&safe_initial]);covered=sum(r<len(sizes) and sizes[r]*R*R>=.15 for r in region_ids)
   distance=np.linalg.norm(cell-np.array(obj['cell']))*R;choices.append(((covered,float(connected.sum())*R*R-.02*distance),cell))
  if not choices:raise RuntimeError(sid+' no parking pose')
  obj['parking_cell']=max(choices,key=lambda x:x[0])[1].tolist()
 final=largest(nav_with(nominal,items,kernels,'nominal',range(len(items))));endpoint_counts=np.bincount(initial_labels[final&safe_initial]);covered=[int(r) for r in region_ids if r<len(endpoint_counts) and endpoint_counts[r]*R*R>=.15]
 if len(covered)<3:raise RuntimeError(sid+' parking lost initial region coverage')
 for j,o in enumerate(items):
  vertices=np.load(objects/(o['asset_id']+'.npz'))['vertices'];o.update(dimensions_m=np.ptp(vertices,axis=0).tolist(),position_m=[*(origin+np.array(o['cell'])[::-1]*R).tolist(),float(fine['z'])+.015],parking_position_m=[*(origin+np.array(o['parking_cell'])[::-1]*R).tolist(),float(fine['z'])+.015],mass_kg=1.,static_friction=1.,dynamic_friction=.9)
 grids=dict(floor=floor,blocked=~static,base_blocked=bands[0],origin=origin,resolution=R,nominal=nominal,optimistic=optimistic,initial_nominal=initial_nom,initial_optimistic=initial_opt,cleared_nominal=final)
 record={k:v for k,v in old.items() if k not in ('objects','endpoint_regions','task_diversity','layout_topology','component_regions','object_connectivity_roles')};record.update(objects=items,physics_validation='pending_new_layout',revision='multiple_bottlenecks',free_area_m2=float(static.sum()*R*R),largest_robot_component_m2=float(nominal.sum()*R*R))
 record,diversity=sample(record,grids,kernels)
 diversity['layout_unchanged']=False
 pairs=Counter(tuple(sorted((int(initial_labels[tuple(t['start_cell'])]),int(initial_labels[tuple(t['goal_cell'])])))) for t in record['tasks'] if t['class']=='interaction_required')
 if len(pairs)<3:raise RuntimeError(sid+' tasks still cover fewer than three initial-region pairs')
 record['layout_topology']={'meaningful_initial_regions':len(region_ids),'covered_initial_regions':covered,'initial_region_areas_m2':{str(int(r)):float(counts[r]*R*R) for r in region_ids},'required_task_region_pairs':{f'{k[0]}-{k[1]}':v for k,v in pairs.items()},'minimum_region_core_area_m2':.65,'minimum_endpoint_space_per_retained_region_m2':.15}
 for t in record['tasks']:t['initial_start_component']=int(initial_labels[tuple(t['start_cell'])]);t['initial_goal_component']=int(initial_labels[tuple(t['goal_cell'])])
 return record,grids

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--fine',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only');ap.add_argument('--resume',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 catalog=json.loads((a.source/'object_catalog.json').read_text());names={o['asset']:o['name'] for o in catalog['objects']};kernels={(aid,yaw):asset_kernels(a.objects/(aid+'.npz'),yaw) for aid in catalog['active_assets'] for yaw in (0,90)}
 for ep in sorted(a.source.glob('S*/episodes.json')):
  old=json.loads(ep.read_text());sid=old['id']
  if a.only and sid!=a.only:continue
  if a.resume and (a.out/sid/'episodes.json').exists():continue
  d,g=design(old,np.load(a.fine/(sid+'.npz')),kernels,names,a.objects);dest=a.out/sid;dest.mkdir(exist_ok=True);dump(dest/'episodes.json',d);np.savez_compressed(dest/'evaluation_grid.npz',**g);print(sid,'READY',json.dumps(d['layout_topology']),flush=True)
 for name in ('object_catalog.json','source_asset_audit.json','environment.json'):shutil.copy2(a.source/name,a.out/name)
if __name__=='__main__':main()
