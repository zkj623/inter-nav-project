"""Inspect composed scene stages and compare authored object poses with frozen layouts."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();results=[]
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text())
 for state,filename,position in [('initial','scene.usda','position_m'),('parked','cleared_scene.usda','parking_position_m')]:
  stage=Usd.Stage.Open(str(ep.parent/filename));assert not stage.GetCompositionErrors(),filename
  scenes=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)];assert scenes==['/World/PhysicsScene'],scenes
  environment=stage.GetPrimAtPath('/World/Environment');meshes=sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(environment,Usd.TraverseInstanceProxies()));assert meshes>10
  for o in d['objects']:
   prim=stage.GetPrimAtPath('/World/Objects/'+o['instance_id']);assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
   actual=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation();assert np.allclose(actual,o[position],atol=1e-6),(d['id'],o['instance_id'],actual)
   assert any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()))
  results.append({'scene':d['id'],'state':state,'environment_meshes':meshes,'object_count':len(d['objects']),'passed':True})
assert len(results)==20
report={'passed':True,'stages':results,'scope':'USD composition, active physics scenes, rigid-body presence, object transforms; not simulation execution'}
(a.out/'usd_validation.json').write_text(json.dumps(report,indent=2));print('20 composed USD stages verified')
