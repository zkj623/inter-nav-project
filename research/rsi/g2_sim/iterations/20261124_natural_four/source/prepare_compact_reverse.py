from pathlib import Path
import json,numpy as np,copy
from scipy.spatial.transform import Rotation,Slerp
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'geometry_lib','exec'))
from posture_anchor import solve_posture
O=R/'compact_reverse';O.mkdir(exist_ok=True)
def mirror(q):
 for i,sg in enumerate([-1,1,-1,1,-1,1,-1]):q[f'idx{61+i}_arm_r_joint{i+1}']=sg*q[f'idx{21+i}_arm_l_joint{i+1}']
 return q
for a in ['stool_wood_four_leg','table_side']:
 d=json.loads((R/'natural_compact'/(a+'.json')).read_text());p=d['plans'][0];plans[a]=p;B=np.array(p['base_pose']);tcp=np.array(p['tcp']);tg=k.fk(p['q_grasp'],B)['arm_l_end_link']@tcp;tp=k.fk(p['q_pre'],B)['arm_l_end_link']@tcp;rot=Slerp([0,1],Rotation.from_matrix([tg[:3,:3],tp[:3,:3]]));last=p['q_grasp'].copy();reverse=[last];ok=True
 for u in np.linspace(.125,1.,8):
  t=tg.copy();t[:3,3]=(1-u)*tg[:3,3]+u*tp[:3,3];t[:3,:3]=rot(u).as_matrix();q,pe,re=solve_posture(k,'l',t,last,B,tcp);q=mirror(q)
  if pe>.002 or re>.02 or not edge(a,last,q):print('PATH_FAIL',a,u,pe,re,flush=True);ok=False;break
  reverse.append(q);last=q
 if not ok:continue
 p['q_pre']=reverse[-1];p['q_contact_approach']=list(reversed(reverse[:-1]));p['q_approach']=[p['q_pre']]
 if not edge(a,p['q_start'],p['q_pre']):
  found=False
  for dz in [.12,.24,.36]:
   t=tp.copy();t[2,3]+=dz;q,pe,re=k.solve('l',t,p['q_pre'],B,tcp);q=mirror(q)
   if pe<.002 and re<.02 and edge(a,p['q_start'],q) and edge(a,q,p['q_pre']):p['q_approach']=[q,p['q_pre']];found=True;break
  if not found:print('HOME_FAIL',a,flush=True);continue
 p['posture_revision']='compact_reverse: plan approach backwards from compact grasp to preserve elbow configuration, mirrored continuous lift and anchored feedback';(O/(a+'.json')).write_text(json.dumps(d,indent=2));print('READY',a,flush=True)
