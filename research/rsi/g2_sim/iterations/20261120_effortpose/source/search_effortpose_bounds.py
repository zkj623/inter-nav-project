from pathlib import Path
import json,numpy as np
from scipy.optimize import minimize
from gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;c=json.loads((R/'roundretry_ground_lift_config.json').read_text());q0=c['default_joint_positions_rad'];base=np.eye(4);base[2,3]=.04;body=[f'idx0{i}_body_joint{i}' for i in range(1,4)];left=[next(n for n in q0 if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in q0 if f'arm_r_joint{i}' in n) for i in range(1,8)];sg=[-1,1,-1,1,-1,1,-1];names=body+left
# Bound independent of arm configuration: torque differences between parallel serial waist axes.
records=[]
for theta1,theta2 in [(-10,30),(-20,40),(-40,80),(-60,120)]:
 vals=[]
 for ar in [q0,c['common_initial_joint_positions_rad']]:
  q={**ar,**dict(zip(body,np.radians([theta1,theta2,-theta1-theta2])))};t=g.torques(q,base);vals.append([t[body[0]]-t[body[1]],t[body[1]]-t[body[2]]])
 records.append(dict(deg=[theta1,theta2,-theta1-theta2],differences=vals));print('DIFFERENCE',records[-1],flush=True)
# Fast cached analytic gravity for all joints, with optional 0.5kg/hand payload.
act=[j for j in k.joints if j['kind']=='revolute' and not j['mimic'] and ('body_joint' in j['name'] or 'arm_' in j['name'])];links=list(g.e.inertia);mass=np.array([g.e.inertia[n][0] for n in links]);points=np.array([g.e.inertia[n][1] for n in links]);mask=np.array([[n in g.e.desc[j['name']] for n in links] for j in act]);zeroR=k.fk(q0,base)['body_link5'][:3,:3];cached={}
def evalstate(x):
 key=tuple(x)
 if key in cached:return cached[key]
 q={**q0,**dict(zip(names,x))};q.update({r:q[l]*s for l,r,s in zip(left,right,sg)});f=k.fk(q,base);pos=np.array([(f[n]@p)[:3] for n,p in zip(links,points)]);out=[]
 for j,m in zip(act,mask):
  t=f[j['parent']]@j['origin'];axis=t[:3,:3]@j['axis'];origin=t[:3,3];mom=np.cross(pos-origin,np.column_stack([np.zeros(len(mass)),np.zeros(len(mass)),-9.81*mass]));tau=float(np.sum((mom@axis)*m));out.append(-tau)
 torso=(f['body_link5'][:3,:3]@zeroR.T)[:,2];tilt=float(np.arccos(np.clip(torso[2],-1,1)));val=(q,f,np.array(out),tilt);cached[key]=val
 if len(cached)>10000:cached.clear();cached[key]=val
 return val
bounds=[(k.by_name[n]['lower']+.003,k.by_name[n]['upper']-.003) for n in names];out=[]
for limit in [50.,40.]:
 for objective in ['shoulder_z','wrist_z','contact_z']:
  def fun(x):
   q,f,t,tilt=evalstate(x);link='arm_l_link2' if objective=='shoulder_z' else 'arm_l_end_link';z=f[link][2,3]
   if objective=='contact_z':z=(f[link]@np.array([-.0045,0,.1532,1]))[2]
   return float(z)
  def cons(x):
   q,f,t,tilt=evalstate(x);caps=np.array([min(limit,j['effort']) for j in act]);return np.r_[caps-t,caps+t,np.radians(20)-tilt]
  best=None
  seeds=[np.array([q0[n] for n in names]),np.array([{**q0,**dict(zip(body,np.radians([-15,30,-15])))}[n] for n in names]),np.array([c['common_initial_joint_positions_rad'][n] for n in names])]
  for seed in seeds:
   fit=minimize(fun,np.clip(seed,[v[0] for v in bounds],[v[1] for v in bounds]),bounds=bounds,constraints=[dict(type='ineq',fun=cons)],method='SLSQP',options=dict(maxiter=180,ftol=1e-9));q,f,t,tilt=evalstate(fit.x);valid=cons(fit.x).min()>-1e-4
   if valid and (best is None or fun(fit.x)<best['minimum_height_m']):best=dict(limit_nm=limit,objective=objective,minimum_height_m=fun(fit.x),q=q,torso_tilt_deg=float(np.degrees(tilt)),body_deg={n:float(np.degrees(q[n])) for n in body},torques_nm={j['name']:float(v) for j,v in zip(act,t)},optimizer_success=bool(fit.success),optimizer_message=fit.message,scope='Optimistic quasi-static candidate only; no self/floor/object collision constraint and no motion feasibility or payload check')
  if best:out.append(best);print('MINIMUM',limit,objective,best['minimum_height_m'],best['body_deg'],best['torso_tilt_deg'],flush=True)
  else:print('NO_SOLUTION',limit,objective,flush=True)
(R/'effortpose_optimistic_bounds.json').write_text(json.dumps(dict(differences=records,results=out),indent=2))
