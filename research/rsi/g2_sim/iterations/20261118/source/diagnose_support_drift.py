from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
for p in R.glob('strategy15_*clock_trials/*/*/*/result.json'):
 d=json.loads(p.read_text());plan=d['plan'];ss=json.loads(p.with_name('samples.json').read_text());hold=[s for s in ss if s['phase']=='hold']
 if not hold:continue
 offsets=[]
 for s in hold:
  base=np.array(plan['base_pose']);base[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],base)
  centers=[]
  for side in ['l','r']:
   local=plan['candidate']['contact_centers_local'][side]
   centers.append((fk[f'arm_{side}_end_link']@np.r_[local,1.])[:3])
  offsets.append(np.array(s['object_position'])-np.mean(centers,axis=0))
 drift=max(np.linalg.norm(x-offsets[0]) for x in offsets)
 print(d['asset'],p.parts[-5],'wrist_frame_drift_mm',d.get('hold_relative_drift_m',0)*1000,'support_midpoint_drift_mm',drift*1000,flush=True)
