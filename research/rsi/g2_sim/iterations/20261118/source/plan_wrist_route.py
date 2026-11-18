"""Preserve frame pinch points, rotate approach about the closing axis."""
from pathlib import Path
import json,numpy as np,subprocess,os
from scipy.spatial.transform import Rotation
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;env=Envelope();k=env.k
out=[];reject={};aid='chair_dining'
source=json.loads((R/'strategy15_chairfar_plans'/(aid+'.json')).read_text())['plans']
for source_index in [0,1]:
 p=source[source_index];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);fk=k.fk(p['q_grasp'],base)
 for angle in [-10,10,-20,20,-30,30]:
  targets={s:fk[f'arm_{s}_end_link']@tcp for s in ['l','r']}
  for side,t in targets.items():t[:3,:3]=t[:3,:3]@Rotation.from_euler('x',angle if side=='l' else -angle,degrees=True).as_matrix()
  def solve(ts,q):
   q=q.copy()
   for side,t in ts.items():
    sol,pe,re=k.solve(side,t,q,base,tcp)
    if pe>.003 or re>.03:return None
    q.update({n:v for n,v in sol.items() if f'arm_{side}_joint' in n})
   if not env.check(q)['ok']:return None
   return q
  grasp=solve(targets,p['q_grasp'])
  if grasp is None:continue
  pretargets={s:t.copy() for s,t in targets.items()}
  for t in pretargets.values():t[:3,3]-=.09*t[:3,2]
  pre=solve(pretargets,grasp)
  if pre is None:continue
  closed={**grasp,'idx31_gripper_l_inner_joint1':0.,'idx71_gripper_r_inner_joint1':0.}
  lifts=[];last=closed
  for dz in np.linspace(.025,.20,8):
   tt={s:t.copy() for s,t in targets.items()}
   for t in tt.values():t[2,3]+=dz
   q=solve(tt,last)
   if q is None:break
   lifts.append(q);last=q
  if len(lifts)<8:continue
  out.append({**p,'q_pre':pre,'q_grasp':grasp,'q_close':closed,'q_lift':lifts,'source_candidate':source_index,'wrist_route_angle_deg':angle,'initial_pose_scope':'low pregrasp; descent not yet validated'})
  print('planned',source_index,angle,flush=True)
folder=R/'strategy15_wrist_plans';folder.mkdir(exist_ok=True)
(folder/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=out),indent=2))
for name in ['validate_plans','audit_native_self','audit_object_clearance']:
 s=(R/f'strategy15_guard_{name}.py').read_text().replace('strategy15_guard','strategy15_wrist')
 path=R/f'strategy15_wrist_{name}.py';path.write_text(s)
 subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(path),aid],cwd=R,check=True)
