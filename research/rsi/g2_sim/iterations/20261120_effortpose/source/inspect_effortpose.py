from pathlib import Path
import json,numpy as np
from gravity_compensation import GravityCompensation
R=Path(__file__).resolve().parent;g=GravityCompensation();k=g.e.k;c=json.loads((R/'roundretry_ground_lift_config.json').read_text());base=np.eye(4);base[2,3]=.04
print('MASS',sum(m for m,_ in g.e.inertia.values()))
for label,q in [('standby',c['default_joint_positions_rad']),('old_low',c['common_initial_joint_positions_rad'])]:
 f=k.fk(q,base);t=g.torques(q,base);print(label,'body_torque',[round(t[n],2) for n in t if 'body_joint' in n],'shoulder',f['arm_l_link2'][:3,3],'wrist',f['arm_l_end_link'][:3,3])
for angles in [(0,0,0),(-20,40,-20),(-40,80,-40),(-50,100,-50),(-60,120,-60),(-60,140,-80)]:
 q={**c['default_joint_positions_rad'],**dict(zip(['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'],np.radians(angles)))};f=k.fk(q,base);t=g.torques(q,base);print(angles,'torques',[round(t[n],1) for n in t if 'body_joint' in n],'shoulder',np.round(f['arm_l_link2'][:3,3],3),'wrist',np.round(f['arm_l_end_link'][:3,3],3))
