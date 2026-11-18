"""Offline overlay: hide rendering only; retain original collision and physics materials."""
import argparse
from pathlib import Path
from pxr import Usd,UsdGeom
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
for scene in sorted(a.out.glob('S*/scene.usda')):
 stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
 root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim());root.GetPrim().GetReferences().AddReference(str(scene.resolve()))
 while True:
  instances=[p for p in stage.Traverse() if p.IsInstance()]
  if not instances:break
  for prim in instances:prim.SetInstanceable(False)
 for prim in list(stage.Traverse()):
  if prim.GetTypeName()=='Shader' or 'Light' in prim.GetTypeName():prim.SetActive(False)
  elif prim.IsA(UsdGeom.Mesh):UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
 flat=Usd.Stage.Open(stage.Flatten())
 for prim in list(Usd.PrimRange.AllPrims(flat.GetPseudoRoot())):
  if prim and (prim.GetTypeName()=='Shader' or 'Light' in prim.GetTypeName()):flat.RemovePrim(prim.GetPath())
 flat.GetRootLayer().Export(str(scene.parent/'physics_only.usdc'))
