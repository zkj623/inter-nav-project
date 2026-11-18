"""Build render-free physics stages, reusing the already audited environment cache."""
import argparse,json
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--old',type=Path);ap.add_argument('--objects',type=Path,required=True);a=ap.parse_args()
cache=a.out/'physics_assets';cache.mkdir(exist_ok=True)
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text())
 for obj in d['objects']:
  target=cache/(obj['asset_id']+'.usdc')
  if target.exists():continue
  stage=Usd.Stage.Open(str(a.objects/(obj['asset_id']+'.usdc')))
  while True:
   instances=[p for p in stage.Traverse() if p.IsInstance()]
   if not instances:break
   for p in instances:p.SetInstanceable(False)
  flat=Usd.Stage.Open(stage.Flatten())
  for p in list(Usd.PrimRange.AllPrims(flat.GetPseudoRoot())):
   if p and (p.GetTypeName()=='Shader' or 'Light' in p.GetTypeName()):flat.RemovePrim(p.GetPath())
   elif p and p.IsA(UsdGeom.Mesh):UsdGeom.Imageable(p).CreateVisibilityAttr('invisible')
  flat.GetRootLayer().Export(str(target))
 for parked,name in [(False,'physics_only.usdc'),(True,'physics_parked.usdc')]:
  if a.old is None:
   # Portable path: flatten this machine's rematerialized source stage.
   stage=Usd.Stage.Open(str(ep.parent/('cleared_scene.usda' if parked else 'scene.usda')))
   while True:
    instances=[p for p in stage.Traverse() if p.IsInstance()]
    if not instances:break
    for p in instances:p.SetInstanceable(False)
   flat=Usd.Stage.Open(stage.Flatten())
   for p in list(Usd.PrimRange.AllPrims(flat.GetPseudoRoot())):
    if p and (p.GetTypeName()=='Shader' or 'Light' in p.GetTypeName()):flat.RemovePrim(p.GetPath())
    elif p and p.IsA(UsdGeom.Mesh):UsdGeom.Imageable(p).CreateVisibilityAttr('invisible')
   flat.GetRootLayer().Export(str(ep.parent/name));continue
  stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
  env=UsdGeom.Xform.Define(stage,'/World/Environment');env.GetPrim().GetReferences().AddReference(str(a.old/d['id']/'physics_only.usdc'),'/World/Environment')
  for o in d['objects']:
   xf=UsdGeom.Xform.Define(stage,'/World/Objects/'+o['instance_id']);xf.GetPrim().GetReferences().AddReference(str(cache/(o['asset_id']+'.usdc')));xf.ClearXformOpOrder();xf.AddTranslateOp(opSuffix='benchmark').Set(Gf.Vec3d(*o['parking_position_m' if parked else 'position_m']));xf.AddRotateZOp(opSuffix='benchmark').Set(float(o['yaw_deg']))
  for p in list(stage.Traverse()):
   if p.IsA(UsdPhysics.Scene):p.SetActive(False)
  phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81);stage.GetRootLayer().Export(str(ep.parent/name))
 print(d['id'],'physics overlays ready',flush=True)
