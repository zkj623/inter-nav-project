from pathlib import Path
import json,isaacsim
from pxr import Usd,UsdPhysics,UsdGeom
R=Path(__file__).resolve().parent;s=Usd.Stage.Open(str(R/'torquelow_direct_trials/torquelow_stooldirect_plans/stool_wood_four_leg/candidate0_repeat1/prepared_scene.usdc'))
for p in s.Traverse():
 if str(p.GetPath()).startswith('/World/TestObject') and p.GetAttribute('physics:rigidBodyEnabled').Get():print('OBJECT_RIGID_DECL',str(p.GetPath()),'HasAPI',p.HasAPI(UsdPhysics.RigidBodyAPI),'schemas',p.GetAppliedSchemas(),'reset',UsdGeom.Xformable(p).GetResetXformStack() if p.IsA(UsdGeom.Xformable) else None)
for file in ['torquelow_direct0.log','torquelow_single0.log']:
 lines=(R/file).read_text(errors='ignore').splitlines();print(file,[(i+1,l[:300]) for i,l in enumerate(lines) if any(x in l.lower() for x in ['nested','resetxform','rigid body of','rigidbodyapi on an instance','rigid body parent'])][:20])
