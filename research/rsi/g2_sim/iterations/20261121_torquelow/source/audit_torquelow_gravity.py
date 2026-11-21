from pathlib import Path
import json
import isaacsim
from pxr import Usd
R=Path(__file__).resolve().parent;f=R/'torquelow_direct_trials/torquelow_stooldirect_plans/stool_wood_four_leg/candidate0_repeat1/prepared_scene.usdc';s=Usd.Stage.Open(str(f));out=[]
for p in s.Traverse():
 if str(p.GetPath()).startswith('/robot') and p.GetAttribute('physics:rigidBodyEnabled').IsValid():out.append(dict(path=str(p.GetPath()),disable_gravity=p.GetAttribute('physxRigidBody:disableGravity').Get(),kinematic_enabled=p.GetAttribute('physics:kinematicEnabled').Get()))
report=dict(stage=str(f),bodies=out,any_robot_gravity_disabled=any(r['disable_gravity'] is True for r in out),any_robot_kinematic=any(r['kinematic_enabled'] is True for r in out));print(json.dumps(report));(R/'torquelow_gravity_flags_audit.json').write_text(json.dumps(report,indent=2));assert not report['any_robot_gravity_disabled'] and not report['any_robot_kinematic']
