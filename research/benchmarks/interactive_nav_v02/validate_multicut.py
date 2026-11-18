"""Audit multiple blocking locations, distinct disconnected-region pairs, and route diversity."""
import argparse,json,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage as ndi
from upgrade_connectivity import asset_kernels,nav_with,stamp,R,STRUCT,dump
from diversify_tasks import corridor,overlap
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);a=ap.parse_args();reports=[];kernels={}
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());g=np.load(ep.parent/'evaluation_grid.npz');clear=ndi.distance_transform_edt(~g['blocked'])*R;core=g['nominal']&(clear>.40);labels,_=ndi.label(g['initial_optimistic'],STRUCT);counts=np.bincount(labels[core]);counts[0]=0;major=np.flatnonzero(counts*R*R>=.65);assert len(major)>=3
 safe=g['initial_nominal']&g['cleared_nominal']&(clear>.40);retained=np.bincount(labels[safe]);recovered=[int(r) for r in major if r<len(retained) and retained[r]*R*R>=.15];assert len(recovered)>=3
 for o in d['objects']:
  key=o['asset_id'],o['yaw_deg']
  if key not in kernels:kernels[key]=asset_kernels(a.objects/(key[0]+'.npz'),key[1])
 # Check object fitting and pairwise separation in both states, using raster footprints.
 for field in ('cell','parking_cell'):
  occupied=np.zeros_like(g['nominal'])
  for o in d['objects']:
   mask=stamp(occupied.shape,kernels[o['asset_id'],o['yaw_deg']]['fit'],o[field]);assert not (mask&g['blocked']).any(),(d['id'],o['instance_id'],field,'static collision');assert not (mask&occupied).any(),(d['id'],field,'object overlap');occupied|=mask
 needed=[t for t in d['tasks'] if t['class']=='interaction_required'];pairs=Counter(tuple(sorted((int(labels[tuple(t['start_cell'])]),int(labels[tuple(t['goal_cell'])])))) for t in needed);assert len(pairs)>=3
 used_objects=set(i for t in needed for i in t['optimistic_relocation_subset']);assert len(used_objects)>=2,(d['id'],'one blocking object dominates every required task')
 used_components=sorted(set(int(labels[tuple(t[k])]) for t in d['tasks'] for k in ('start_cell','goal_cell')));display={r:f'C{j+1:02}' for j,r in enumerate(used_components)};regions=[]
 for r in used_components:
  pp=np.argwhere(safe&(labels==r));assert len(pp)>0;center=pp[np.argmin(np.linalg.norm(pp-pp.mean(0),axis=1))];regions.append({'component_id':r,'name':display[r],'center_xy_m':(g['origin']+center[::-1]*R).tolist(),'safe_endpoint_area_m2':float(len(pp)*R*R)})
 for t in d['tasks']:
  t['initial_start_component']=int(labels[tuple(t['start_cell'])]);t['initial_goal_component']=int(labels[tuple(t['goal_cell'])]);t['initial_start_region']=display[t['initial_start_component']];t['initial_goal_region']=display[t['initial_goal_component']]
 roles=[]
 for j,o in enumerate(d['objects']):
  one,_=ndi.label(nav_with(g['optimistic'],d['objects'],kernels,'optimistic',[j]),STRUCT);others,_=ndi.label(nav_with(g['optimistic'],d['objects'],kernels,'optimistic',[i for i in range(len(d['objects'])) if i!=j]),STRUCT)
  opened=[];still_blocked=[]
  for t in needed:
   s,q=tuple(t['start_cell']),tuple(t['goal_cell'])
   if one[s] and one[s]==one[q]:opened.append(t['id'])
   if not others[s] or others[s]!=others[q]:still_blocked.append(t['id'])
  role={'instance_id':o['instance_id'],'asset_id':o['asset_id'],'tasks_opened_by_moving_this_object_only':opened,'tasks_still_blocked_when_all_other_objects_are_parked':still_blocked,'tasks_using_this_object_in_selected_witness':[t['id'] for t in needed if j in t['optimistic_relocation_subset']]};roles.append(role)
 paths=[corridor(np.rint((np.array(t['reference_route_xy_m'])-g['origin'])[:,::-1]/R).astype(int)) for t in d['tasks']];ovs=[overlap(x,y) for x,y in itertools.combinations(paths,2)];assert max(ovs)<=.65+1e-9
 bins=Counter(t['length_band'] for t in needed);assert set(bins)=={'short','medium','long'} and max(bins.values())<=4
 endpoints=[tuple(t[k]) for t in d['tasks'] for k in ('start_cell','goal_cell')];assert max(Counter(endpoints).values())<=2
 d['component_regions']=regions;d['object_connectivity_roles']=roles;d['task_diversity']['layout_unchanged']=False
 d['layout_topology'].update(required_task_region_pairs={f'{display[p[0]]}-{display[p[1]]}':n for p,n in pairs.items()},objects_used_in_required_tasks=[d['objects'][j]['instance_id'] for j in sorted(used_objects)],task_component_count=len(used_components))
 dump(ep,d);reports.append({'scene':d['id'],'meaningful_initial_regions':len(major),'retained_initial_regions':len(recovered),'task_component_count':len(used_components),'distinct_required_component_pairs':len(pairs),'objects_used_in_required_tasks':len(used_objects),'unique_endpoint_positions':len(set(endpoints)),'required_length_bands':dict(bins),'maximum_route_corridor_iou':max(ovs),'mean_route_corridor_iou':float(np.mean(ovs)),'object_roles':roles,'passed':True});print(d['id'],len(major),'regions',len(pairs),'pairs',len(used_objects),'objects',flush=True)
assert len(reports)==10
dump(a.out/'layout_validation.json',{'passed':True,'scenes':reports,'scope':'projected raster topology and object footprint checks; dynamic G2 interaction remains untested'})
dump(a.out/'diversity_validation.json',{'passed':True,'scenes':reports,'layout_revision':'multiple_bottlenecks','corridor_iou_limit':.65})
dump(a.out/'task_diversity.json',{'passed':True,'scenes':[json.loads(ep.read_text())['task_diversity'] for ep in sorted(a.out.glob('S*/episodes.json'))],'layout_changes':10,'route_corridor_iou_limit':.65})
