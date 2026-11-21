from pathlib import Path
import json,numpy as np
from gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;c=json.loads((R/'roundretry_ground_lift_config.json').read_text());q=c['common_initial_joint_positions_rad'];torques=g.torques(q);gradient={}
for n in [f'idx0{i}_body_joint{i}' for i in range(1,6)]:
 e=1e-6;num=(g.potential({**q,n:q[n]+e})-g.potential({**q,n:q[n]-e}))/(2*e);gradient[n]=dict(analytic_nm=torques[n],potential_gradient_nm=num,error_nm=abs(torques[n]-num))
assert max(x['error_nm'] for x in gradient.values())<1e-4
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':6,'physics_gpu':6,'multi_gpu':False})
try:
 from pxr import Usd,UsdPhysics,UsdGeom
 f=R/'torquelow_reference_trials/torquelow_reference_plans/bin_round_lidded/candidate0_repeat1/prepared_scene.usdc';stage=Usd.Stage.Open(str(f));rigid={p.GetName():p for p in stage.Traverse() if str(p.GetPath()).startswith('/robot') and p.HasAPI(UsdPhysics.RigidBodyAPI)};fk=k.fk({});parent={j['child']:j for j in k.joints};groups={n:[] for n in rigid}
 for n,(m,p) in g.e.inertia.items():
  b=n
  while b not in groups:
   j=parent[b];assert j['kind']=='fixed';b=j['parent']
  local=np.linalg.inv(fk[b])@fk[n]@p;groups[b].append((m,local[:3]))
 mass_total=0.;errors=[]
 for n,p in rigid.items():
  api=UsdPhysics.MassAPI(p);actualmass=float(api.GetMassAttr().Get());actualcom=np.array(api.GetCenterOfMassAttr().Get());em=sum(m for m,_ in groups[n]);ec=sum(m*v for m,v in groups[n])/em;mass_total+=actualmass;errors.append(dict(link=n,mass_error_kg=abs(actualmass-em),com_error_m=float(np.linalg.norm(actualcom-ec))))
 assert max(x['mass_error_kg'] for x in errors)<1e-4 and max(x['com_error_m'] for x in errors)<1e-6
 limits=[]
 for p in stage.Traverse():
  if p.IsA(UsdPhysics.RevoluteJoint) and 'body_joint' in p.GetName():
   api=UsdPhysics.RevoluteJoint(p);limits.append(dict(name=p.GetName(),lower_deg=float(api.GetLowerLimitAttr().Get()),upper_deg=float(api.GetUpperLimitAttr().Get()),initial_drive_maxforce=float(UsdPhysics.DriveAPI(p,'angular').GetMaxForceAttr().Get())))
 report=dict(verified=True,gravity_potential_gradient=gradient,native_robot_mass_kg=mass_total,source_urdf_mass_kg=sum(m for m,_ in g.e.inertia.values()),max_link_mass_error_kg=max(x['mass_error_kg'] for x in errors),max_link_com_error_m=max(x['com_error_m'] for x in errors),stage_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),waist_limits=limits,scope='Pinned original crs URDF and corrected-inertia USD consistency; does not verify physical robot CAD/hardware');(R/'torquelow_model_audit.json').write_text(json.dumps(report,indent=2));print('MODEL_AUDIT_OK',json.dumps(report),flush=True)
finally:app.close()
