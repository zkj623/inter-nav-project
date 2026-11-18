"""Optional compatibility shading overlay; original scene and physics stay referenced."""
import argparse,json
from pathlib import Path
from pxr import Usd,UsdGeom
from material_fallback import apply_koostruct_material_fallbacks
from mdl_preview_materials import convert_exported_pbr_materials
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
for source in sorted(a.out.glob('S*/scene.usda')):
 stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim());root.GetPrim().GetReferences().AddReference(str(source.resolve()))
 while True:
  instances=[p for p in stage.Traverse() if p.IsInstance()]
  if not instances:break
  for p in instances:p.SetInstanceable(False)
 report={'koo':apply_koostruct_material_fallbacks(stage,'/World'),'pbr':convert_exported_pbr_materials(stage,'/World'),'scope':'preview shading approximation, original texture files retained; physics unchanged'}
 stage.GetRootLayer().Export(str(source.parent/'preview.usda'));(source.parent/'preview_conversion.json').write_text(json.dumps(report,indent=2));print(source.parent.name,report,flush=True)
