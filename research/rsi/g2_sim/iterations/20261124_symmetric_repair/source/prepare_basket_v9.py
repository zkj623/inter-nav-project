from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps')
import trimesh
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;O=R/'repair_v9';O.mkdir(exist_ok=True);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
d=json.loads((R/'repair_v8/basket_rectangular.json').read_text());p=d['plans'][0];base=np.array(p['base_pose']);tcp=np.array(p['tcp']);v=np.load(R/'object_assets/basket_rectangular.npz');mesh=trimesh.Trimesh(v['vertices'],v['faces'],process=False);shifts={}
for side,label in [('l','left'),('r','right')]:
 point=p['candidate'][label]['pose'];x=point[0][3];z=point[2][3];hits,_,_=mesh.ray.intersects_location([[x,-.3,z]],[[0,1,0]],multiple_hits=True);ys=sorted(hits[:,1]);pair=ys[-2:] if side=='l' else ys[:2];mid=float(np.mean(pair));shifts[side]=mid-point[1][3];point[1][3]=mid;p['candidate'][label]['width']=float(abs(pair[1]-pair[0]));print(side,'wall adjustment',shifts[side])
def solve(q):
 fk=k.fk(q,base);out=q.copy()
 for side in ['l','r']:
  target=fk[f'arm_{side}_end_link']@tcp;target[1,3]+=shifts[side];out,pe,re=k.solve(side,target,out,base,tcp);assert pe<.0015 and re<.015,(side,pe,re)
 return out
for key in ['q_pre','q_grasp','q_close']:p[key]=solve(p[key])
p['q_lift']=[solve(q) for q in p['q_lift']];p['q_approach']=[p['q_pre'].copy()];p['repair_revision']='v9 ray-cast actual curved wall center at new grasp x; correct5.6mm outward offset; bounded jaw aperture and0.8Nm cap'
(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2));print('BASKET_V9_READY')
