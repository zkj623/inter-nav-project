from pathlib import Path
from pxr import Usd
import json
R=Path(__file__).resolve().parent;rows=[]
files=[R/'robot_assets/g2_contact_inertia_fixed.usdc']+list(R.glob('commonfix_solver_trials/**/prepared_scene.usdc'))+list(R.glob('commonfix_accurate_trials/**/prepared_scene.usdc'))
for f in files:
 s=Usd.Stage.Open(str(f));props=[]
 for p in s.Traverse():
  for at in p.GetAttributes():
   if 'solverPositionIterationCount' in at.GetName() or 'solverVelocityIterationCount' in at.GetName():props.append((str(p.GetPath()),at.GetName(),at.Get()))
  if p.GetName()=='PrecisionCollision':
   rel=p.GetRelationship('material:binding:physics');props.append((str(p.GetPath()),'physics_material',[str(v) for v in rel.GetTargets()]))
 row=dict(scene=str(f),properties=props);rows.append(row);print(json.dumps(row),flush=True)
(R/'commonfix_scene_review.json').write_text(json.dumps(rows,indent=2))
