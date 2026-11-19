from pathlib import Path
import ast
R=Path(__file__).resolve().parent
for a in ['basket_rectangular','case_handheld']:
 s=(R/f'commonfix_precise_{a}_trial.py').read_text().replace("ROOT/'commonfix_precise_trials'","ROOT/'commonfix_accurate_trials'")
 needle="        cooking=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())"
 s=s.replace(needle,"        from pxr import UsdShade\n        material=UsdShade.Material(stage.GetPrimAtPath(str(link.GetPath())+'/ContactMaterial'))\n        assert material.GetPrim().IsValid()\n        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')\n"+needle)
 s=s.replace("    result['precision_collision_experiment']=","    result['precision_material_preserved']=True\n    result['precision_collision_experiment']=")
 ast.parse(s);(R/f'commonfix_accurate_{a}_trial.py').write_text(s)
