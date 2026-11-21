from pathlib import Path
import json,numpy as np,itertools,time,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from gravity_compensation import GravityCompensation
from liftall_collision import space,robot
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;cfg=json.loads((R/'roundretry_ground_lift_config.json').read_text());default=cfg['default_joint_positions_rad'];low=cfg['common_initial_joint_positions_rad'];base=np.eye(4);base[2,3]=.04
names=[n for n in default if ('body_joint' in n or 'head_joint' in n or 'arm_' in n)];body=[n for n in names if 'body_joint' in n];left=[next(n for n in names if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in names if f'arm_r_joint{i}' in n) for i in range(1,8)];act=[k.by_name[n] for n in names];links=list(g.e.inertia);mass=np.array([g.e.inertia[n][0] for n in links]);ipoints=np.array([g.e.inertia[n][1] for n in links]);mask=np.array([[n in g.e.desc[j['name']] for n in links] for j in act],float);submass=mask@mass;caps=np.array([50. if 'body_joint' in j['name'] else j['effort']*.9 for j in act]);bounds=[(k.by_name[n]['lower']+.02,k.by_name[n]['upper']-.02) for n in names];torsozero=k.fk(default,base)['body_link5'][:3,:3];tcp=np.array([-.0045,0,.1532,1.]);aabbnames=[n for n in robot if n!=k.root and not n.startswith('chassis')];aabbcenter=np.array([robot[n].bounds.mean(0) for n in aabbnames]);aabbextent=np.array([(robot[n].bounds[1]-robot[n].bounds[0])/2 for n in aabbnames]);payloadmasks=np.array([[f'arm_{s}_end_link' in g.e.desc[j['name']] for s in ('l','r')] for j in act],float)
def run(task):
 mode,theta_deg,tilt_limit,seedidx=task;theta=np.radians(theta_deg);forward=np.array([np.cos(theta),np.sin(theta),0.]);side=np.array([-np.sin(theta),np.cos(theta),0.]);center=.48*forward;cache={};localbounds=bounds
 def state(x):
  key=tuple(x)
  if key in cache:return cache[key]
  q={**default,**dict(zip(names,x)),'idx31_gripper_l_inner_joint1':-.6,'idx71_gripper_r_inner_joint1':-.6};f=k.fk(q,base);positions=np.array([(f[n]@p)[:3] for n,p in zip(links,ipoints)]);jt=np.array([f[j['parent']]@j['origin'] for j in act]);orig=jt[:,:3,3];axes=np.array([t[:3,:3]@j['axis'] for t,j in zip(jt,act)]);weighted=mask@(mass[:,None]*positions)-submass[:,None]*orig;cp=np.array([(f[f'arm_{s}_end_link']@tcp)[:3] for s in ('l','r')]);payload=np.array([.5,.5]) if mode=='dual_free' else np.array([1.,0.]);weighted+=payloadmasks@(payload[:,None]*cp)-(payloadmasks@payload)[:,None]*orig;tau=9.81*(axes[:,0]*weighted[:,1]-axes[:,1]*weighted[:,0]);tilt=float(np.arccos(np.clip((f['body_link5'][:3,:3]@torsozero.T)[2,2],-1,1)));ts=np.array([f[n] for n in aabbnames]);z=ts[:,2,3]+np.einsum('ni,ni->n',ts[:,2,:3],aabbcenter)-np.einsum('ni,ni->n',abs(ts[:,2,:3]),aabbextent);ret=q,f,cp,tau,tilt,z;cache[key]=ret
  if len(cache)>8000:cache.clear();cache[key]=ret
  return ret
 def objective(x):
  q,f,cp,tau,tilt,z=state(x);return float(cp[0,2]+(0.00005*np.sum((x-np.array([default[n] for n in names]))**2)))
 def eq(x):
  q,f,cp,tau,tilt,z=state(x)
  if mode=='dual_free':return np.r_[cp[0,:2]-(center+.11*side)[:2],cp[1,:2]-(center-.11*side)[:2],cp[0,2]-cp[1,2]]
  a=cp[0,:2]-center[:2]
  if mode=='single_top':return np.r_[a,f['arm_l_end_link'][:2,2],f['arm_l_end_link'][2,2]+1.]
  return a
 def inequality(x):
  q,f,cp,tau,tilt,z=state(x);return np.r_[caps-tau,caps+tau,np.radians(tilt_limit)-tilt,z-.005,cp[0,2]-.025]
 seed=default.copy() if seedidx==0 else low.copy();rng=np.random.default_rng(20261121+seedidx)
 if seedidx==1:
  seed.update(idx01_body_joint1=-.22,idx02_body_joint2=.45,idx03_body_joint3=-1.5)
  for n in left+right:seed[n]=.5*default[n]+.5*low[n]
 if seedidx==2:
  seed.update(idx01_body_joint1=-.1,idx02_body_joint2=.18,idx03_body_joint3=-1.8)
  for n in right:seed[n]=float(rng.uniform(k.by_name[n]['lower']+.15,k.by_name[n]['upper']-.15))
 seed['idx05_body_joint5']=float(np.clip(theta,-2.8,2.8));x0=np.clip([seed[n] for n in names],[a for a,b in bounds],[b for a,b in bounds]);start=time.monotonic();fit=minimize(objective,x0,bounds=localbounds,constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=inequality)],method='SLSQP',options=dict(maxiter=200,ftol=2e-7));q,f,cp,tau,tilt,z=state(fit.x);valid=max(abs(eq(fit.x)))<.003 and min(inequality(fit.x))>-.003;hits=intersections(q) if valid else None;proxy=space['collision'](q) if valid else None
 return dict(task=task,valid_static=bool(valid),geometry_clear=bool(valid and not hits and not proxy),contact_height_m=float(cp[0,2]),contacts_xyz_m=cp.tolist(),q=q,torso_tilt_deg=float(np.degrees(tilt)),waist_torques_nm={n:float(tau[names.index(n)]) for n in body},torques_with_1kg_nm=dict(zip(names,map(float,tau))),min_floor_aabb_gap_m=float(z.min()),self_intersections=hits,native_collision=proxy,eq_error=float(max(abs(eq(fit.x)))),min_inequality=float(min(inequality(fit.x))),optimizer_success=bool(fit.success),message=fit.message,seconds=time.monotonic()-start)
if __name__=='__main__':
 tasks=[(m,t,tilt,s) for m,t,tilt,s in itertools.product(['single_free','dual_free'],[0,90,180],[60,110],[0,1,2])];rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(run,t) for t in tasks]
  for f in as_completed(fs):
   try:r=f.result()
   except Exception as e:print('WORKER_ERROR',repr(e),flush=True);continue
   rows.append(r);print('RESULT',r['task'],'static',r['valid_static'],'clear',r['geometry_clear'],'z',round(r['contact_height_m'],4),'tilt',round(r['torso_tilt_deg'],1),r['message'],flush=True);(R/'torquelow_search_results.json').write_text(json.dumps(rows,indent=2))
 print('SEARCH_DONE',len(rows),'VALID',sum(r['valid_static'] for r in rows),'CLEAR',sum(r['geometry_clear'] for r in rows),flush=True)
