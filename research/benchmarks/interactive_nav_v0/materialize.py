"""Recreate the frozen suite with locally supplied asset roots, without resampling tasks."""
import argparse,json,os,hashlib,shutil
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf
ap=argparse.ArgumentParser();ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
if a.out.exists():raise FileExistsError('Use a fresh output directory')
validation=json.loads((a.suite/'geometry_validation.json').read_text())
audit=a.suite/'source_asset_audit.json'
if audit.exists():
 for name,sha in json.loads(audit.read_text()).get('scene_entry_sha256',{}).items():
  if hashlib.sha256((a.home/name).read_bytes()).hexdigest()!=sha:raise ValueError('Scene entry changed: '+name)
for aid,sha in validation['source_usdc_sha256'].items():
 path=a.objects/(aid+'.usdc')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Object asset changed: '+aid)
for f in a.suite.glob('S*/episodes.json'):
 d=json.loads(f.read_text())
 if not (a.home/d['scene']).is_file():raise FileNotFoundError(a.home/d['scene'])
a.out.mkdir(parents=True)
for f in sorted(a.suite.glob('S*/episodes.json')):
 d=json.loads(f.read_text());dest=a.out/d['id'];dest.mkdir();d['physics_validation']='pending_in_this_workspace';(dest/f.name).write_text(json.dumps(d,indent=2,ensure_ascii=False))
 for name in ('evaluation_grid.npz','layout.png'):
  if (f.parent/name).exists():shutil.copy2(f.parent/name,dest/name)
 stage=Usd.Stage.CreateNew(str(dest/'scene.usda'));UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
 base=UsdGeom.Xform.Define(stage,'/World/Environment');base.GetPrim().GetReferences().AddReference(os.path.relpath(a.home/d['scene'],dest));base.AddScaleOp().Set(Gf.Vec3d(.01))
 for p in list(stage.Traverse()):
  if p.IsA(UsdPhysics.Scene):p.SetActive(False)
 for item in d['objects']:
  xf=UsdGeom.Xform.Define(stage,'/World/Objects/'+item['instance_id']);xf.GetPrim().GetReferences().AddReference(os.path.relpath(a.objects/(item['asset_id']+'.usdc'),dest));xf.ClearXformOpOrder();xf.AddTranslateOp(opSuffix='benchmark').Set(Gf.Vec3d(*item['position_m']));xf.GetPrim().SetCustomDataByKey('benchmark:asset_id',item['asset_id'])
 phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81);stage.GetRootLayer().Save()
 print(d['id'],flush=True)
