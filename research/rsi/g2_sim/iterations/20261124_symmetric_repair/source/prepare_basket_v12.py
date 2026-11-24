from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps')
import trimesh
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v12';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
d=json.loads((R/'repair_v11/basket_rectangular.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);v=np.load(R/'object_assets/basket_rectangular.npz');m=trimesh.Trimesh(v['vertices'],v['faces'],process=False);shifts={}
for side,label in [('l','left'),('r','right')]:
 pose=p['candidate'][label]['pose'];pose[2][3]-=.01;hits,_,_=m.ray.intersects_location([[pose[0][3],-.3,pose[2][3]]],[[0,1,0]],multiple_hits=True);ys=sorted(hits[:,1]);pair=ys[-2:] if side=='l' else ys[:2];mid=float(np.mean(pair));shifts[side]=mid-pose[1][3];pose[1][3]=mid
def solve(q):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  t=fk[f'arm_{side}_end_link']@tcp;t[1,3]+=shifts[side];t[2,3]-=.01;out,pe,re=k.solve(side,t,out,base,tcp);assert pe<.0015 and re<.015,(side,pe,re)
 return out
for key in ['q_pre','q_grasp','q_close']:p[key]=solve(p[key])
p['q_lift']=[solve(q) for q in p['q_lift']];p['q_approach']=[p['q_pre'].copy()];p['repair_revision']='v12 10mm deeper actual-wall pinch, avoid top-lip ledge; centered yaw0 and2Nm bounded pinch with slow lift';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2));print('BASKET_V12_READY')
