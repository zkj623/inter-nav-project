from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;d=json.loads((R/'narrowrest_search.json').read_text());c=d['candidates'][0];q=c['q'];k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
print('BASE',d['base_bounds']['width_m'],'PREVIOUS_ARMS',d['previous_arm_bounds']['width_m'],'NARROW_ARMS',c['visual_arm_bounds']['width_m'])
print('OUTER_LINKS',sorted(c['visual_arm_bounds']['links'].items(),key=lambda v:-max(abs(v[1][0]),abs(v[1][1])))[:12])
print('LEFT_WRIST',c['wrist_left_xyz_m'],'LEFT_ELBOW',c['elbow_left_xyz_m'],'MIN_ARM_MARGIN_DEG',c['minimum_arm_margin_deg'])
print('JOINTS_DEG',{n:float(np.degrees(v)) for n,v in q.items() if 'arm_' in n})
for name in ['arm_l_link1','arm_l_link2']:
 vals=[];mesh=np.load(R/'robot_assets'/(name+'.npz'))['vertices']
 joint=next(n for n in q if 'arm_l_joint1' in n)
 for a in np.linspace(k.by_name[joint]['lower']+.18,k.by_name[joint]['upper']-.18,21):
  fk=k.fk({**q,joint:float(a)})[name];vals.append(float((mesh@fk[:3,:3].T+fk[:3,3])[:,1].max()))
 print('SHOULDER_AXIS_SWEEP',name,min(vals),max(vals))
