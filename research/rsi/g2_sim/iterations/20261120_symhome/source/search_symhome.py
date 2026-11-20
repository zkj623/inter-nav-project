from pathlib import Path
import json,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,space
from audit_self_collision import intersections
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;cfg=json.loads((R/'ground_lift_training_config.json').read_text());old=cfg['common_initial_joint_positions_rad'];fixed=cfg['fixed_joint_positions_rad'];env=Envelope();base=np.eye(4);base[2,3]=.04
default={n:0. for n in old}
for side,prefix in [('l','2'),('r','6')]:
 for j,d in [(2,-45),(4,-75),(6,-55)]:default[next(n for n in old if f'arm_{side}_joint{j}' in n)]=float(np.radians(d))
left=[next(n for n in old if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in old if f'arm_r_joint{i}' in n) for i in range(1,8)]
test={**default,**dict(zip(left,[.2,-.7,.3,-1.2,.4,-.8,.1]))};f=k.fk(test,base);best=None
for signs in itertools.product([-1,1],repeat=7):
 q={**test,**{r:test[l]*s for l,r,s in zip(left,right,signs)}};ff=k.fk(q,base)
 err=sum(np.linalg.norm(ff[f'arm_r_link{i}'][:3,3]-ff[f'arm_l_link{i}'][:3,3]*[1,-1,1]) for i in range(2,8))
 if best is None or err<best[0]:best=(err,signs)
assert best[0]<1e-5,best
signs=best[1];print('MIRROR',best,flush=True)
def mirror(q):return {**q,**{r:q[l]*s for l,r,s in zip(left,right,signs)}}
def clear(q):
 if any(q.get(n,0)<j['lower']-1e-6 or q.get(n,0)>j['upper']+1e-6 for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic']):return False
 if space['collision'](q) or intersections(q):return False
 fk=k.fk(q,base)
 for n,g in robot.items():
  if 'chassis' in n or n==k.root:continue
  if (g.bounds.mean(0)@fk[n][:3,:3].T+fk[n][:3,3])[2]-.5*(np.abs(fk[n][:3,:3])@(g.bounds[1]-g.bounds[0]))[2]<.005:return False
 return env.check(q)['ok']
def audit_edge(a,b):
 count=max(3,int(max(abs(a.get(n,0)-b.get(n,0)) for n in old)/.035)+2)
 for u in np.linspace(0,1,count):
  q={n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in old}
  if not clear(q):return False
 return True
print('DEFAULT_CLEAR',clear(default),env.check(default),flush=True)
lo=np.array([k.by_name[n]['lower']+.12 for n in left]);hi=np.array([k.by_name[n]['upper']-.12 for n in left]);ref=k.fk({**default,**fixed},base)['arm_l_end_link'][:3,:3]
out=[]
for x,y,z,ey in itertools.product([.54,.60],[.24,.28],[.66,.72],[.34,.38]):
 def residual(a):
  q={**old,**dict(zip(left,a))};fk=k.fk(q,base);w=fk['arm_l_end_link'];e=fk['arm_l_link4'][:3,3]
  return np.r_[(w[:3,3]-[x,y,z])*6,(e[1]-ey)*6,(e[0]-.43)*1.5,Rotation.from_matrix(ref.T@w[:3,:3]).as_rotvec()*.06,.02*(a-np.array([default[n] for n in left]))]
 fit=least_squares(residual,np.clip([default[n] for n in left],lo,hi),bounds=(lo,hi),max_nfev=160)
 q=mirror({**old,**dict(zip(left,fit.x)),**{n:-.785 for n in old if 'gripper' in n}})
 if not clear(q):continue
 fk=k.fk(q,base);el=fk['arm_l_link4'][:3,3];wr=fk['arm_l_end_link'][:3,3]
 # Keep arms at the selected folded rest shape while lowering body; if required raise wrist first at default torso.
 upright={**q,**{n:default[n] for n in fixed}}
 routes=[[q],[upright,q]]
 route=next((r for r in routes if all(audit_edge(a,b) for a,b in zip([default]+r,r))),None)
 if route is None:continue
 row=dict(q=q,waypoints=route,elbow_span_m=float(2*el[1]),wrist_xyz_left=wr.tolist(),static=env.check(q),requested=[x,y,z,ey])
 out.append(row);print('ACCEPT',row['elbow_span_m'],wr.tolist(),len(route),flush=True)
out.sort(key=lambda r:abs(r['elbow_span_m']-.72)+.5*abs(r['wrist_xyz_left'][0]-.58))
(R/'symhome_search.json').write_text(json.dumps(dict(default_source='vendor/genie_sim_drive_audit/.../scene_flat_g2_op.yaml init_joint_pos degrees; omitted joints=0; base z0.04 pinned as existing test scope',default=default,mirror_signs=signs,candidates=out),indent=2));print('DONE',len(out),flush=True)
