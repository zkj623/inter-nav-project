"""Validate frozen layouts and all 100 tasks, independently of task selection."""
import argparse,json,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from upgrade_connectivity import asset_kernels,nav_with,graph,R,STRUCT

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);a=ap.parse_args();rows=[];assets=set();all_tasks=[];kernels={}
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());grid=np.load(ep.parent/'evaluation_grid.npz');origin=grid['origin'];items=d['objects'];assets.update(o['asset_id'] for o in items)
 for o in items:
  key=(o['asset_id'],o['yaw_deg'])
  if key not in kernels:kernels[key]=asset_kernels(a.objects/(key[0]+'.npz'),key[1])
 opt=nav_with(grid['optimistic'],items,kernels,'optimistic');initial=nav_with(grid['nominal'],items,kernels,'nominal');final=nav_with(grid['nominal'],items,kernels,'nominal',range(len(items)))
 assert np.array_equal(opt,grid['initial_optimistic']);assert np.array_equal(initial,grid['initial_nominal'])
 labels,_=ndi.label(opt,STRUCT);basegraph,basecells,baseindex=graph(grid['nominal']);cleared_labels,_=ndi.label(final,STRUCT)
 counts={name:sum(t['class']==name for t in d['tasks']) for name in ['interaction_required','no_interaction_optimal']};assert counts=={'interaction_required':8,'no_interaction_optimal':2},(d['id'],counts)
 assert len({tuple(sorted((tuple(t['start_cell']),tuple(t['goal_cell'])))) for t in d['tasks']})==10
 fixed_hash=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest();checked=[]
 for t in d['tasks']:
  s=tuple(t['start_cell']);g=tuple(t['goal_cell']);xy=np.array(t['reference_route_xy_m']);cells=np.rint((xy-origin)[:,::-1]/R).astype(int)
  assert tuple(cells[0])==s and tuple(cells[-1])==g
  assert initial[s] and initial[g];step=np.diff(cells,axis=0);assert (np.abs(step)<=1).all() and (np.abs(step).sum(1)>0).all()
  length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());assert abs(length-t['cleared_route_length_m'])<1e-6
  assert length>=5.0-1e-9,(t['id'],'reference route shorter than 5 m',length)
  permitted=final if t['class']=='interaction_required' else initial
  assert permitted[tuple(cells.T)].all(),t['id']
  for cell,delta in zip(cells[:-1],step):
   if delta[0] and delta[1]:assert permitted[cell[0]+delta[0],cell[1]] and permitted[cell[0],cell[1]+delta[1]],t['id']
  if t['class']=='interaction_required':
   assert labels[s] and labels[g] and labels[s]!=labels[g],t['id']
   # A 0.5 m goal tolerance must not make an initially disconnected goal reachable.
   reachable=labels==labels[s];rr,cc=np.nonzero(reachable);gap=float(np.hypot(rr-g[0],cc-g[1]).min()*R);assert gap>t['goal_tolerance_m'],(t['id'],'goal tolerance bypass',gap)
   assert cleared_labels[s] and cleared_labels[s]==cleared_labels[g]
   minimum=None
   for n in range(1,len(items)+1):
    for subset in itertools.combinations(range(len(items)),n):
     ll,_=ndi.label(nav_with(grid['optimistic'],items,kernels,'optimistic',subset),STRUCT)
     if ll[s] and ll[s]==ll[g]:minimum=n;break
    if minimum is not None:break
   assert minimum==t['minimum_relocations_in_optimistic_model'] and minimum>0
   witness=t['optimistic_relocation_subset'];assert len(witness)==minimum
   wl,_=ndi.label(nav_with(grid['optimistic'],items,kernels,'optimistic',witness),STRUCT);assert wl[s] and wl[s]==wl[g]
  else:
   assert labels[s]==labels[g] and labels[s]>0
   minimum=0;gap=None;distance=dijkstra(basegraph,indices=int(baseindex[s]));assert abs(float(distance[baseindex[g]])-length)<1e-6
  checked.append({'task':t['id'],'class':t['class'],'reference_route_m':length,'minimum_relocations':minimum,'initial_reachable_set_distance_to_goal_m':gap,'passed':True});all_tasks.append(checked[-1])
 rows.append({'scene':d['id'],'fixed_layout_sha256':fixed_hash,'objects':len(items),'tasks':checked,'passed':True})
assert len(rows)==10 and len(assets)==15 and len(all_tasks)==100
report={'version':'0.2.0','passed':True,'scenes':rows,'scene_count':10,'layout_count':10,'task_count':100,'object_type_count':15,'interaction_required':80,'no_interaction_optimal':20,'multi_relocation_tasks':sum(t['minimum_relocations']>=2 for t in all_tasks),'required_path_range_m':[min(t['reference_route_m'] for t in all_tasks if t['class']=='interaction_required'),max(t['reference_route_m'] for t in all_tasks if t['class']=='interaction_required')],'scope':'2.5 cm projected raster navigation model; initial optimistic radius 0.25 m, nominal radius 0.36 m; not a complete G2 body or manipulation feasibility proof','robot_rollout_validated':False}
(a.out/'geometry_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
