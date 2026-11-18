import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from build import mask_objects,pathfind,length,footprint
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);a=ap.parse_args()
counts=collections.Counter();report=[];sources={}
for f in sorted(a.out.glob('S*/episodes.json')):
 data=json.loads(f.read_text());grid=np.load(f.parent/'evaluation_grid.npz');free=grid['floor'].astype(bool)&~grid['blocked'].astype(bool);mask=mask_objects(free,data['objects']);stage=Usd.Stage.Open(str(f.parent/'scene.usda'))
 assert len(data['tasks'])==6 and len(data['objects'])==3
 assert len({(tuple(t['start_position_m']),tuple(t['goal_position_m'])) for t in data['tasks']})==6
 for x in data['objects']:
  counts[x['asset_id']]+=1;p=stage.GetPrimAtPath('/World/Objects/'+x['instance_id']);assert p and p.HasAPI(UsdPhysics.RigidBodyAPI)
  assert np.all(free[footprint(free.shape,x['cell'],np.array(x['dimensions_m'][:2])/2+.08)])
  assert any(c.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get() for c in Usd.PrimRange(p))
  pos=UsdGeom.XformCache().GetLocalToWorldTransform(p).ExtractTranslation();assert np.allclose(pos,x['position_m'])
  asset=a.objects/(x['asset_id']+'.usdc');sources[x['asset_id']]=hashlib.sha256(asset.read_bytes()).hexdigest()
  mesh=np.load(a.objects/(x['asset_id']+'.npz'));v=mesh['vertices'];assert np.allclose(np.ptp(v,axis=0),x['dimensions_m'],atol=.015),x['asset_id']
 for t in data['tasks']:
  assert t['reference_path_m']>=3
  cells=[tuple(np.rint((np.array(t[key][:2])-grid['origin'])/.1).astype(int)[::-1]) for key in ('start_position_m','goal_position_m')]
  assert all(mask[c] for c in cells)
  l=length(pathfind(mask,*cells));assert (l is None)==(t['initial_detour_path_m'] is None)
  if l is not None:assert abs(l-t['initial_detour_path_m'])<1e-6
 report.append({'id':data['id'],'area_m2':data['floor_area_m2'],'tasks':6,'classes':dict(collections.Counter(t['geometric_class'] for t in data['tasks']))})
assert len(report)==10 and len(counts)==15 and set(counts.values())=={2}
result={'passed':True,'scene_count':10,'task_count':60,'object_type_count':15,'object_instance_count':30,'asset_usage':dict(counts),'source_usdc_sha256':sources,'scenes':report,'scope':'USD composition, rigid bodies, active colliders, dimensions, endpoint validity and detour recalculation; no robot rollout'}
(a.out/'geometry_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
