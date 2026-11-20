from pathlib import Path
import json,collections
from pxr import Usd,UsdPhysics
R=Path(__file__).resolve().parent;f=R/'symhome_v4_trials/symhome_plans/basket_rectangular/candidate0_repeat1/prepared_scene.usdc';stage=Usd.Stage.Open(str(f));groups=collections.Counter()
for p in stage.Traverse():
 if not p.HasAPI(UsdPhysics.CollisionAPI):continue
 if 'gripper_' not in str(p.GetPath()) and not str(p.GetPath()).startswith('/World/TestObject'):continue
 groups[(('object' if str(p.GetPath()).startswith('/World/TestObject') else 'finger'),str(p.GetAttribute('physxCollision:contactOffset').Get()),str(p.GetAttribute('physxCollision:restOffset').Get()))]+=1
print(groups)
for prefix in ['symhome_v4_trials','symhome_v4precise_trials']:
 f=R/prefix/'symhome_plans/basket_rectangular/candidate0_repeat1/result.json'
 if f.exists():
  d=json.loads(f.read_text());print(prefix,{k:d.get(k) for k in ['failure','simulated_seconds','max_contact_surface_error_m','prepared_max_tracking_error_rad']})
