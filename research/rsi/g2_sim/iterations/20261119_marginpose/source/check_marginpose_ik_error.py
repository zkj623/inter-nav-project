from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');inventory={x['asset']:x for x in json.loads((R/'fixedpose_inventory.json').read_text())}
for aid in ['box_reinforced','case_handheld','stool_rectangular']:
 old=inventory[aid]['plan'];new=json.loads((R/'marginpose_selected7_validated_plans'/(aid+'.json')).read_text())['plans'][0];shift=np.array(new['object_position'])-old['object_position'];rows=[]
 for key in ['q_pre','q_grasp','q_close','q_lift']:
  a=old.get(key,old['q_grasp']);b=new.get(key,new['q_grasp']);a=a if isinstance(a,list) else [a];b=b if isinstance(b,list) else [b]
  for i,(qa,qb) in enumerate(zip(a,b)):
   fa=k.fk(qa,np.array(old['base_pose']));fb=k.fk(qb,np.array(new['base_pose']))
   for side in ['l','r']:
    x,y=fa[f'arm_{side}_end_link'],fb[f'arm_{side}_end_link'];rows.append(dict(stage=key,index=i,side=side,error_mm=float(1000*np.linalg.norm(y[:3,3]-x[:3,3]-shift)),error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(x[:3,:3].T@y[:3,:3]).as_rotvec())))))
 print(aid,'max',max(x['error_mm'] for x in rows),'grasp',[x for x in rows if x['stage']=='q_grasp'],flush=True)
