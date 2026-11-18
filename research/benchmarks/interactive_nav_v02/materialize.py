"""Recreate frozen initial/parked USD stages using user-supplied asset directories."""
import argparse,json,shutil,os
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf
ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('fixtures'));ap.add_argument('--home',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for name in ('object_catalog.json','coverage.json','task_diversity.json','diversity_validation.json','layout_validation.json','selected.json'):
 source=a.fixtures/name
 if source.exists():shutil.copy2(source,a.out/name)
(a.out/'physics_validation.json').write_text(json.dumps({'passed':False,'scenes':[],'scope':'not run on this machine'},indent=2))
for ep in sorted(a.fixtures.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());dest=a.out/d['id'];dest.mkdir(exist_ok=True)
 assert (a.home/d['scene']).is_file(),a.home/d['scene']
 for o in d['objects']:assert (a.objects/(o['asset_id']+'.usdc')).is_file(),o['asset_id']
 for f in ep.parent.iterdir():
  if f.suffix in ('.json','.npz','.png') and f.name!='physics_smoke.json':shutil.copy2(f,dest/f.name)
 d['physics_validation']='not_run_on_this_machine';(dest/'episodes.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
 for parked,filename in [(False,'scene.usda'),(True,'cleared_scene.usda')]:
  target=dest/filename
  if target.exists():stage=Usd.Stage.Open(str(target));stage.GetRootLayer().Clear()
  else:stage=Usd.Stage.CreateNew(str(target))
  UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim());env=UsdGeom.Xform.Define(stage,'/World/Environment');env.GetPrim().GetReferences().AddReference(os.path.relpath(a.home/d['scene'],dest));env.AddScaleOp().Set(Gf.Vec3d(.01))
  for p in list(stage.Traverse()):
   if p.IsA(UsdPhysics.Scene):p.SetActive(False)
  for o in d['objects']:
   xf=UsdGeom.Xform.Define(stage,'/World/Objects/'+o['instance_id']);xf.GetPrim().GetReferences().AddReference(os.path.relpath(a.objects/(o['asset_id']+'.usdc'),dest));xf.ClearXformOpOrder();xf.AddTranslateOp(opSuffix='benchmark').Set(Gf.Vec3d(*o['parking_position_m' if parked else 'position_m']));xf.AddRotateZOp(opSuffix='benchmark').Set(float(o['yaw_deg']))
  phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81);stage.GetRootLayer().Save()
 print(d['id'],'materialized',flush=True)
