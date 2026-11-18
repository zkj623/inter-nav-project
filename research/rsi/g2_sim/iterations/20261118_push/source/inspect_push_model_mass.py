from pathlib import Path
from pxr import Usd,UsdPhysics
import json,math
R=Path(__file__).resolve().parent;rows=[]
for aid in ['chair_dining','chair_wheeled']:
 stage=Usd.Stage.Open(str(R/'object_assets'/(aid+'.usdc')))
 p=stage.GetPrimAtPath('/Object');a=UsdPhysics.MassAPI(p)
 raw=list(a.GetCenterOfMassAttr().Get())
 rows.append(dict(asset=aid,mass=a.GetMassAttr().Get(),center_of_mass=[v if math.isfinite(v) else str(v) for v in raw],center_of_mass_mode='automatic USD sentinel; runtime COM not measured',diagonal_inertia=list(a.GetDiagonalInertiaAttr().Get()),inertia_mode='automatic zero sentinel',rigid_bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)],joints=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]))
(R/'push_model_mass.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
