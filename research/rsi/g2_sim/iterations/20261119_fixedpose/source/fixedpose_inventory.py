from pathlib import Path
import json,numpy as np
from g2_kinematics import Kinematics,transform
R=Path(__file__).resolve().parent
archive=R.parent/'g2-publish-20261118/research/rsi/g2_sim'
baselines=json.loads((archive/'baselines.json').read_text())
for rel in ['iterations/20261118','iterations/20261118_support']:
 baselines.update(json.loads((archive/rel/'baselines.json').read_text()))
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');rows=[]
for aid,b in baselines.items():
 if aid=='stool_small':continue
 pd=b.get('plan_dir',b.get('plan_directory'));print('BASE',aid,b,flush=True)
 if not pd:continue
 p=json.loads((R/pd/(aid+'.json')).read_text())['plans'][b['candidate']]
 q=p['q_pre'];body={n:q.get(n,0.) for n in k.by_name if 'body_joint' in n};fk=k.fk(q,transform((0,0,.04)))
 row=dict(asset=aid,baseline=b,plan=p,body=body,shoulder_height=float(fk['arm_l_link1'][2,3]))
 rows.append(row)
 print('POSE',aid,np.round(np.degrees(list(body.values())),2).tolist(),row['shoulder_height'],flush=True)
(R/'fixedpose_inventory.json').write_text(json.dumps(rows,indent=2))
