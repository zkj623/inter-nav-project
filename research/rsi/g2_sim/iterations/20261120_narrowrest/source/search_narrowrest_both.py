from pathlib import Path
import json,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from liftall_collision import k,space
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;d=json.loads((R/'restpose_search.json').read_text());old=d['candidates'][0]['q'];home=d['work_home'];left=[next(n for n in old if f'arm_l_joint{i}' in n) for i in range(1,8)];right=[next(n for n in old if f'arm_r_joint{i}' in n) for i in range(1,8)];signs=d['mirror_signs'];base=np.eye(4);base[2,3]=.04
verts={p.stem:np.load(p)['vertices'] for p in (R/'robot_assets').glob('*.npz') if p.stem in k.links};arms=[n for n in verts if n.startswith('arm_') or n.startswith('gripper_')];baseparts=[n for n in verts if n==k.root or n.startswith('chassis_')];lf=[n for n in arms if n.startswith('arm_l_') or n.startswith('gripper_l_')]
def bounds(q,names):
 f=k.fk(q,base);out={n:[float((verts[n]@f[n][:3,:3].T+f[n][:3,3])[:,1].min()),float((verts[n]@f[n][:3,:3].T+f[n][:3,3])[:,1].max())] for n in names};return dict(width_m=max(b[1] for b in out.values())-min(b[0] for b in out.values()),min_y_m=min(b[0] for b in out.values()),max_y_m=max(b[1] for b in out.values()),links=out)
b= bounds(old,baseparts);original=bounds(old,arms);limit=min(-b['min_y_m'],b['max_y_m']);print('BASE',b['width_m'],'OLD_ARMS',original['width_m'],flush=True)
def mirror(q):return {**q,**{r:q[l]*s for l,r,s in zip(left,right,signs)}}
def clear(q):return not space['collision'](q) and not intersections(q)
def edge(a,b):
 return all(clear({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in old}) for u in np.linspace(0,1,max(3,int(max(abs(a.get(n,0)-b.get(n,0)) for n in old)/.05)+2)))
lo=np.array([k.by_name[n]['lower']+.18 for n in left]);hi=np.array([k.by_name[n]['upper']-.18 for n in left]);first=json.loads((R/'narrowrest_search.json').read_text())['candidates'][0]['q'];seed=np.clip([first[n] for n in left],lo+1e-6,hi-1e-6);reference=k.fk(old,base)['arm_l_end_link'][:3,:3];accepted=[];attempts=[]
for ex,wy in itertools.product([.12,.18,.24],[.25,.28,.30]):
 def fun(a):
  q=mirror({**old,**dict(zip(left,a))});f=k.fk(q,base);w=f['arm_l_end_link'];e=f['arm_l_link4'][:3,3];ys=np.array([float(np.max(np.abs((verts[n]@f[n][:3,:3].T+f[n][:3,3])[:,1]))) for n in arms])
  return np.r_[(w[:3,3]-[max(.16,ex),wy,.85])*4,(e-[ex,.265,1.12])*3,np.maximum(0,ys-(limit-.003))*35,Rotation.from_matrix(reference.T@w[:3,:3]).as_rotvec()*.08,.025*(a-seed)]
 fit=least_squares(fun,seed,bounds=(lo,hi),max_nfev=160);q=mirror({**old,**dict(zip(left,fit.x))});f=k.fk(q,base);w=f['arm_l_end_link'][:3,3];e=f['arm_l_link4'][:3,3];valid=clear(q);route=None
 if valid and edge(q,home):route=[home]
 if valid and route is None:
  upper={**home,**{n:0. for n in home if 'body_joint' in n}}
  if clear(upper) and edge(q,upper) and edge(upper,home):route=[upper,home]
 width=bounds(q,arms);row=dict(elbow_target_x=ex,wrist_target_y=wy,clear=valid,route=bool(route),arms_width_m=width['width_m']);attempts.append(row);print(row,flush=True)
 if route and w[2]+.12<e[2] and e[2]+.12<f['arm_l_link2'][2,3]:
  accepted.append(dict(q=q,waypoints=route,score=width['width_m'],visual_arm_bounds=width,wrist_left_xyz_m=w.tolist(),elbow_left_xyz_m=e.tolist(),minimum_arm_margin_deg=float(min(np.degrees(min(q[n]-k.by_name[n]['lower'],k.by_name[n]['upper']-q[n])) for n in left+right))))
 if accepted and width['min_y_m']>=b['min_y_m']+.002 and width['max_y_m']<=b['max_y_m']-.002:break
assert accepted,'No clear narrower standby found';accepted.sort(key=lambda r:(max(0.,b['min_y_m']-r['visual_arm_bounds']['min_y_m'],r['visual_arm_bounds']['max_y_m']-b['max_y_m']),r['score']))
(R/'narrowrest_search_v2.json').write_text(json.dumps(dict(label='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL',mirror_signs=signs,work_home=home,previous_standby=old,base_bounds=b,previous_arm_bounds=original,candidates=accepted,attempts=attempts),indent=2));print('DONE',len(accepted),accepted[0]['score'],flush=True)
