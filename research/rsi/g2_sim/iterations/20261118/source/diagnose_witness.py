from pathlib import Path
import json,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from plan_grasps import mesh
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;old=R.parent/'g2-migration-20261111'
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
p=old/'strategy15_guard_trials/strategy15_guard_validated_plans/stool_rectangular/candidate1_repeat1'
d=json.loads((p/'result.json').read_text());ss=json.loads((p/'samples.json').read_text())
om=mesh('stool_rectangular');om.apply_scale(1000)
worst=[]
for s in ss:
 if s['phase'] not in ['lift','rejected']:continue
 base=np.array(d['plan']['base_pose']);base[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],base)
 for w in s['contact_witnesses']:
  n=w['link'];points=np.array(w['points_world']);fi=(points-fk[n][:3,3])@fk[n][:3,:3];oi=np.array(w['points_object'])
  fm=mesh(n);fm.apply_scale(1000)
  _,fd,_=trimesh.proximity.closest_point(fm,fi*1000);_,od,_=trimesh.proximity.closest_point(om,oi*1000)
  worst.append(dict(t=s['t'],link=n,object_mm=float(od.max()),finger_fk_estimate_mm=float(fd.max())))
print(json.dumps(sorted(worst,key=lambda x:max(x['object_mm'],x['finger_fk_estimate_mm']),reverse=True)[:8],indent=2))
