from pathlib import Path
import json,numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from gravity_compensation import GravityCompensation
from audit_self_collision import intersections
from liftall_collision import space,robot
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;c=json.loads((R/'roundretry_ground_lift_config.json').read_text());q0=c['default_joint_positions_rad'];base=np.eye(4);base[2,3]=.04;body=[f'idx0{i}_body_joint{i}' for i in range(1,4)];left=[next(n for n in q0 if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in q0 if f'arm_r_joint{i}' in n) for i in range(1,8)];names=body+left;sg=[-1,1,-1,1,-1,1,-1];p=json.loads((R/'directprep_plans/bin_round_lidded.json').read_text())['plans'][0];ref=k.fk(p['q_close'],base)['arm_l_end_link'][:3,:3];tcp=np.array(p['candidate']['contact_centers_local']['l']);zeroR=k.fk(q0,base)['body_link5'][:3,:3];act=[j for j in k.joints if j['kind']=='revolute' and not j['mimic'] and ('body_joint' in j['name'] or 'arm_' in j['name'])];cached={}
def evaluate(x):
 key=tuple(x)
 if key in cached:return cached[key]
 q={**q0,**dict(zip(names,x)),**{'idx31_gripper_l_inner_joint1':-.6,'idx71_gripper_r_inner_joint1':-.6}};q.update({r:q[l]*s for l,r,s in zip(left,right,sg)});f=k.fk(q,base);tau=g.torques(q,base);cp={s:(f[f'arm_{s}_end_link']@np.r_[p['candidate']['contact_centers_local'][s],1.])[:3] for s in ('l','r')}
 for j in act:
  t=f[j['parent']]@j['origin'];axis=t[:3,:3]@j['axis']
  for s in ('l','r'):
   if f'arm_{s}_end_link' in g.e.desc[j['name']]:tau[j['name']]-=float(np.dot(np.cross(cp[s]-t[:3,3],[0,0,-4.905]),axis))
 tilt=float(np.arccos(np.clip((f['body_link5'][:3,:3]@zeroR.T)[2,2],-1,1)));ret=q,f,tau,cp,tilt;cached[key]=ret;return ret
bounds=[(k.by_name[n]['lower']+.04,k.by_name[n]['upper']-.04) for n in names];results=[]
for limit in [40.,50.]:
 for x_target in [.40,.45]:
  def fun(x):return float(evaluate(x)[3]['l'][2])
  def eq(x):
   q,f,t,cp,tilt=evaluate(x);return np.r_[cp['l'][:2]-[x_target,.12],Rotation.from_matrix(ref.T@f['arm_l_end_link'][:3,:3]).as_rotvec()]
  def ine(x):
   q,f,t,cp,tilt=evaluate(x);caps=np.array([limit if 'body_joint' in j['name'] else j['effort']*.8 for j in act]);ts=np.array([t[j['name']] for j in act]);return np.r_[caps-ts,caps+ts,np.radians(20)-tilt]
  best=None
  for seed in [q0,p['q_close'],c['common_initial_joint_positions_rad']]:
   x0=np.clip([seed[n] for n in names],[a for a,b in bounds],[b for a,b in bounds]);fit=minimize(fun,x0,bounds=bounds,constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=ine)],method='SLSQP',options=dict(maxiter=180,ftol=1e-8));q,f,t,cp,tilt=evaluate(fit.x);valid=max(abs(eq(fit.x)))<.002 and min(ine(fit.x))>-.001
   if valid:
    hits=intersections(q);native=space['collision'](q);floor=min(float((f[n][:3,3]+f[n][:3,:3]@geo.bounds.mean(0))[2]-(abs(f[n][:3,:3])@((geo.bounds[1]-geo.bounds[0])/2))[2]) for n,geo in robot.items() if n!=k.root and not n.startswith('chassis'))
    row=dict(limit_nm=limit,x_target=x_target,contact_height_m=fun(fit.x),q=q,body_deg={n:float(np.degrees(q[n])) for n in body},torques_with_1kg_payload_nm=t,torso_tilt_deg=float(np.degrees(tilt)),self_intersections=hits,native_collision=native,visible_aabb_floor_gap_m=floor,geometry_clear=not hits and not native and floor>.005,optimizer_success=bool(fit.success))
    if best is None or row['contact_height_m']<best['contact_height_m']:best=row
  if best:results.append(best);print('WORK_CANDIDATE',limit,x_target,best['contact_height_m'],best['body_deg'],'clear',best['geometry_clear'],flush=True)
  else:print('NO_WORK_SOLUTION',limit,x_target,flush=True)
(R/'effortpose_work_candidates.json').write_text(json.dumps(results,indent=2))
