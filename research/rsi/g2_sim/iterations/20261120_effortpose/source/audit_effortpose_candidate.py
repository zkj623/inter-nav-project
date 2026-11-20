from pathlib import Path
import json,numpy as np
from gravity_compensation import GravityCompensation
from audit_self_collision import intersections
from liftall_collision import space
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;c=json.loads((R/'roundretry_ground_lift_config.json').read_text());rows=json.loads((R/'effortpose_work_candidates.json').read_text());best=next(x for x in rows if x['limit_nm']==40 and x['x_target']==.4);q0=c['default_joint_positions_rad'];q=best['q'];body=[f'idx0{i}_body_joint{i}' for i in range(1,4)];record=[]
for u in np.linspace(0,1,81):
 t={n:float((1-u)*q0[n]+u*q[n]) for n in q0};tau=g.torques(t);mx=max(abs(tau[n]) for n in body);hits=intersections(t);proxy=space['collision'](t);record.append(dict(fraction=float(u),max_waist_torque_nm=mx,self_intersections=hits,native_collision=proxy))
print('SELECTED',best['body_deg'],'loadedbody',{n:best['torques_with_1kg_payload_nm'][n] for n in body},'armsdeg',{n:round(np.degrees(v),2) for n,v in q.items() if 'arm_' in n},flush=True);print('PATH',max(x['max_waist_torque_nm'] for x in record),'bad',sum(bool(x['self_intersections'] or x['native_collision']) for x in record),flush=True)
for angles in [(-12,24,-12),(-15,30,-15)]:
 t={**q0,**dict(zip(body,np.radians(angles)))};tau=g.torques(t);print('MILD',angles,{n:tau[n] for n in body},'self',intersections(t),flush=True)
(R/'effortpose_candidate_path.json').write_text(json.dumps(dict(selected=best,empty_path=record,empty_static_path_within50=all(x['max_waist_torque_nm']<=50 for x in record),geometry_path_clear=all(not x['self_intersections'] and not x['native_collision'] for x in record)),indent=2))
