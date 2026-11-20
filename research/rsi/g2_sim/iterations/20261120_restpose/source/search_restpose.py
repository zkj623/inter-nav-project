from pathlib import Path
import json,itertools,numpy as np
from scipy.optimize import least_squares
from liftall_collision import k,robot,space
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;previous=json.loads((R/'symhome_search.json').read_text());home=previous['candidates'][0]['q'];upright={n:0. for n in home};left=[next(n for n in home if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in home if f'arm_r_joint{i}' in n) for i in range(1,8)];signs=previous['mirror_signs'];base=np.eye(4);base[2,3]=.04
def mirror(q):return {**q,**{r:q[l]*s for l,r,s in zip(left,right,signs)}}
def clear(q):
 if any(q.get(n,0)<j['lower']-1e-6 or q.get(n,0)>j['upper']+1e-6 for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic']):return False
 if space['collision'](q) or intersections(q):return False
 fk=k.fk(q,base)
 return all((g.bounds.mean(0)@fk[n][:3,:3].T+fk[n][:3,3])[2]-.5*(np.abs(fk[n][:3,:3])@(g.bounds[1]-g.bounds[0]))[2]>.005 for n,g in robot.items() if 'chassis' not in n and n!=k.root)
def edge(a,b):
 for u in np.linspace(0,1,max(3,int(max(abs(a.get(n,0)-b.get(n,0)) for n in home)/.04)+2)):
  if not clear({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in home}):return False
 return True
lo=np.array([k.by_name[n]['lower']+.18 for n in left]);hi=np.array([k.by_name[n]['upper']-.18 for n in left]);seed=np.clip([previous['default'][n] for n in left],lo+1e-6,hi-1e-6);rows=[]
for x,y,z in itertools.product([.08,.15,.22],[.30,.34,.38],[.78,.86,.94]):
 def residual(a):
  q={**upright,**dict(zip(left,a))};f=k.fk(q,base);w=f['arm_l_end_link'][:3,3];e=f['arm_l_link4'][:3,3]
  return np.r_[(w-[x,y,z])*7,(e-[.03,.32,1.13])*2,.025*(a-seed)]
 fit=least_squares(residual,seed,bounds=(lo,hi),max_nfev=180)
 q=mirror({**upright,**dict(zip(left,fit.x)),**{n:-.3 for n in home if 'gripper' in n}})
 if not clear(q):continue
 f=k.fk(q,base);w=f['arm_l_end_link'][:3,3];e=f['arm_l_link4'][:3,3];shoulder=f['arm_l_link2'][:3,3]
 if np.linalg.norm(w-[x,y,z])>.035 or not (shoulder[2]-.12>e[2]>w[2]+.12):continue
 route=[home] if edge(q,home) else None
 if route is None:
  upper={**home,**{n:0. for n in home if 'body_joint' in n}}
  if clear(upper) and edge(q,upper) and edge(upper,home):route=[upper,home]
 if route is None:continue
 score=abs(w[0]-.12)+abs(w[1]-.33)+abs(w[2]-.86)+.2*len(route)
 rows.append(dict(q=q,waypoints=route,score=float(score),wrist_left_xyz_m=w.tolist(),elbow_left_xyz_m=e.tolist(),shoulder_left_xyz_m=shoulder.tolist(),elbow_span_m=float(2*e[1]),minimum_arm_margin_deg=float(min(np.degrees(min(q[n]-k.by_name[n]['lower'],k.by_name[n]['upper']-q[n])) for n in left+right))))
 print('ACCEPT',rows[-1]['wrist_left_xyz_m'],rows[-1]['elbow_span_m'],len(route),flush=True)
rows.sort(key=lambda r:r['score']);assert rows,'No collision-free arms-down standby found'
(R/'restpose_search.json').write_text(json.dumps(dict(label='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL',work_home=home,candidates=rows),indent=2));print('DONE',len(rows),flush=True)
