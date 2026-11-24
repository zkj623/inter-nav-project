from pathlib import Path
import json,numpy as np,hashlib,ast
from scipy.spatial.transform import Rotation,Slerp
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'geometry_lib','exec'))
from posture_anchor import solve_posture
O=R/'compact_paths';O.mkdir(exist_ok=True)
def mirror(x):
 for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):x[f'idx{61+i}_arm_r_joint{i+1}']=sg*x[f'idx{21+i}_arm_l_joint{i+1}']
 return x
for aid in ['stool_wood_four_leg','table_side']:
 d=json.loads((R/'natural_compact'/(aid+'.json')).read_text());p=d['plans'][0];plans[aid]=p;B=np.array(p['base_pose']);tcp=np.array(p['tcp']);t0=k.fk(p['q_pre'],B)['arm_l_end_link']@tcp;t1=k.fk(p['q_grasp'],B)['arm_l_end_link']@tcp;rot=Slerp([0,1],Rotation.from_matrix([t0[:3,:3],t1[:3,:3]]));last=p['q_pre'].copy();steps=[];ok=True
 for u in np.linspace(.125,1,8):
  t=t0.copy();t[:3,3]=(1-u)*t0[:3,3]+u*t1[:3,3];t[:3,:3]=rot(u).as_matrix();q,pe,re=solve_posture(k,'l',t,last,B,tcp);q=mirror(q)
  if pe>.002 or re>.02 or not edge(aid,last,q):ok=False;print('CONTACT_PATH_FAIL',aid,u,pe,re,flush=True);break
  steps.append(q);last=q
 if not ok:continue
 p['q_contact_approach']=steps;p['q_grasp']=steps[-1];p['q_close']={**steps[-1],'idx31_gripper_l_inner_joint1':0.,'idx71_gripper_r_inner_joint1':0.}
 if not edge(aid,p['q_start'],p['q_pre']):
  found=False
  for dz in [.12,.24,.36]:
   target=t0.copy();target[2,3]+=dz;q,pe,re=k.solve('l',target,p['q_pre'],B,tcp);q=mirror(q)
   if pe<.002 and re<.02 and edge(aid,p['q_start'],q) and edge(aid,q,p['q_pre']):p['q_approach']=[q,p['q_pre'].copy()];found=True;break
  if not found:print('HOME_PATH_FAIL',aid,flush=True);continue
 p['posture_revision']+='; collision-screened high approach waypoint and Cartesian contact approach';(O/(aid+'.json')).write_text(json.dumps(d,indent=2));print('READY',aid,flush=True)
 f=R/f'realpose_{aid}_trial.py';s=f.read_text();old="            segment(qpre,plan['q_grasp'],3.,'approach')";new="""            approach_last=qpre
            approach_points=plan.get('q_contact_approach',[plan['q_grasp']])
            for approach_next in approach_points:
                segment(approach_last,approach_next,3./len(approach_points),'approach');approach_last=approach_next"""
 assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);f.write_text(s);data=f.read_bytes();(R/'source_versions'/(hashlib.sha256(data).hexdigest()+'.py')).write_bytes(data)
