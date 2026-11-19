from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');old=R.parent/'g2-commonstart-20261119';out=[]
for a in ['case_handheld','basket_rectangular']:
 f=old/f'commonstart_slow_trials/commonstart_plans/{a}/candidate0_repeat1/samples.json';ss=json.loads(f.read_text());errors=[]
 for s in ss:
  q=s['joint_positions'];pairs=[]
  for n,j in k.by_name.items():
   m=j['mimic']
   if m and n in q and m['joint'] in q:pairs.append((abs(q[n]-q[m['joint']]*float(m.get('multiplier',1))-float(m.get('offset',0))),n))
  if pairs:
   e,n=max(pairs);errors.append(dict(t=s['t'],phase=s['phase'],error_rad=e,joint=n))
 row=dict(asset=a,max_error=max(errors,key=lambda x:x['error_rad']),median_contact_error_rad=float(np.median([x['error_rad'] for x in errors if x['phase'] in ['lift','contact_refine','close']])));out.append(row);print(row,flush=True)
(R/'commonfix_mimic_review.json').write_text(json.dumps(out,indent=2))
