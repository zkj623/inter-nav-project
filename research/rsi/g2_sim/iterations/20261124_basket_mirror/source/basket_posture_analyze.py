from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');p=json.loads((R/'repair_final_plans/basket_rectangular.json').read_text())['plans'][0]
names={s:[n for n in k.by_name if f'arm_{s}_joint' in n and k.by_name[n]['kind']=='revolute'] for s in ['l','r']};sign=np.array([-1,1,-1,1,-1,1,-1])
for key in ['q_start','q_pre','q_grasp','q_close','q_lift']:
 q=p[key][-1] if key=='q_lift' else p[key];fk=k.fk(q,np.array(p['base_pose']))
 print(key)
 for s in ['l','r']:
  t=fk[f'arm_{s}_end_link'];print(s,'deg',np.round(np.degrees([q[n] for n in names[s]]),1).tolist(),'pos',t[:3,3].round(4).tolist(),'rot',t[:3,:3].round(3).tolist())
 print('mirror_delta_deg',np.round(np.degrees(np.array([q[n] for n in names['r']])-sign*np.array([q[n] for n in names['l']])),1).tolist())
print('tcp',p['tcp'],'mode',p['mode'],'candidate',p['candidate'])
