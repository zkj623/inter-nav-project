"""Regression check against source USD geometry for S08's missing support strips."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from pxr import Usd,UsdPhysics
import scan
from floor_support import augment_floor
from upgrade_connectivity import largest,R,dump

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fine',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);a=ap.parse_args()
d=json.loads((a.out/'S08/episodes.json').read_text());cached=np.load(a.fine/'S08.npz');grid=np.load(a.out/'S08/evaluation_grid.npz');scan.R=R
original,origin,z=scan.floor(a.home/d['scene']);stage=Usd.Stage.Open(str(a.home/d['scene']));corrected,evidence=augment_floor(stage,original,origin,z,R)
assert np.array_equal(corrected,cached['floor']) and np.array_equal(corrected,grid['floor'])
assert np.all(corrected[original.astype(bool)])
obstacles=cached['obstacle_bands'].any(0)
assert np.array_equal(grid['blocked'],~(corrected&~obstacles))
before=largest(ndi.distance_transform_edt(original.astype(bool)&~obstacles)*R>=.36)
after=largest(ndi.distance_transform_edt(corrected&~obstacles)*R>=.36)
assert np.array_equal(after,grid['nominal']);assert after.sum()>before.sum()*3
assert np.array_equal(ndi.distance_transform_edt(corrected&~cached['obstacle_bands'][0].astype(bool))*R>=.25,grid['optimistic'])
affected=[];new_endpoint_tasks=[];added=corrected&~original.astype(bool)
for t in d['tasks']:
 cells=np.rint((np.asarray(t['reference_route_xy_m'])-origin)[:,::-1]/R).astype(int)
 if added[tuple(cells.T)].any():affected.append(t['id'])
 if not before[tuple(t['start_cell'])] or not before[tuple(t['goal_cell'])]:new_endpoint_tasks.append(t['id'])
assert affected and len(new_endpoint_tasks)>=5
for row in evidence['meshes']:
 prim=stage.GetPrimAtPath(row['mesh_path']);ancestors=[]
 while prim and not prim.IsPseudoRoot():
  if prim.HasAPI(UsdPhysics.CollisionAPI):ancestors.append(str(prim.GetPath()))
  prim=prim.GetParent()
 row['collision_api_on_mesh_or_ancestors']=ancestors
 assert ancestors,row['mesh_path']+' missing collision API'
report={'passed':True,'scene':'S08','cause':'semantic-ground-only extraction omitted actual coplanar architectural surfaces categorized as Base/other','previous_selected_robot_center_area_m2':float(before.sum()*R*R),'corrected_selected_robot_center_area_m2':float(after.sum()*R*R),'minimum_reference_route_m':min(t['cleared_route_length_m'] for t in d['tasks']),'maximum_reference_route_m':max(t['cleared_route_length_m'] for t in d['tasks']),'tasks_crossing_restored_floor':affected,'tasks_with_endpoint_outside_previous_component':new_endpoint_tasks,'nominal_radius_m':.36,'source_obstacles_removed':False,'source_usd_changed':False,'evidence':evidence,'scope':'source geometry and raster regression; not a robot rollout'}
dump(a.out/'floor_support_validation.json',report);print(json.dumps({k:v for k,v in report.items() if k!='evidence'},indent=2))
