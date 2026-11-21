from pathlib import Path
import json,math
import isaacsim
from pxr import Usd,UsdGeom,UsdPhysics
R=Path(__file__).resolve().parent;f=R/'torquelow_direct_trials/torquelow_stooldirect_plans/stool_wood_four_leg/candidate0_repeat1/prepared_scene.usdc';stage=Usd.Stage.Open(str(f));report=dict(stage=str(f),meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),object_bodies=[],waist_joints=[],gravity_scenes=[])
for p in stage.Traverse():
 path=str(p.GetPath())
 if path.startswith('/World/TestObject') and p.HasAPI(UsdPhysics.RigidBodyAPI):
  report['object_bodies'].append(dict(path=path,mass_kg=p.GetAttribute('physics:mass').Get(),center_of_mass=list(p.GetAttribute('physics:centerOfMass').Get()) if p.GetAttribute('physics:centerOfMass').Get() is not None and all(math.isfinite(v) for v in p.GetAttribute('physics:centerOfMass').Get()) else None,kinematic=p.GetAttribute('physics:kinematicEnabled').Get(),disable_gravity=p.GetAttribute('physxRigidBody:disableGravity').Get()))
 if 'body_joint' in p.GetName():
  report['waist_joints'].append(dict(name=p.GetName(),lower_deg=p.GetAttribute('physics:lowerLimit').Get(),upper_deg=p.GetAttribute('physics:upperLimit').Get(),initial_drive_max_force=p.GetAttribute('drive:angular:physics:maxForce').Get()))
 if p.GetAttribute('physics:gravityMagnitude').IsValid():report['gravity_scenes'].append(dict(path=path,magnitude=p.GetAttribute('physics:gravityMagnitude').Get(),direction=list(p.GetAttribute('physics:gravityDirection').Get())))
(R/'torquelow_inputs_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
