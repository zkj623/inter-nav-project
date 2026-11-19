"""Retarget verified contact paths to one torso; no object geometry changes."""
from pathlib import Path
import json,sys,copy,time
import numpy as np
from g2_kinematics import Kinematics,transform
from ideal_constraints import Envelope
from fixedpose_collision import check as collision_check
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope()
rows=json.loads((R/'fixedpose_inventory.json').read_text())
profiles=[[-55,150,-80,0,0],[-55,145,-75,0,0],[-60,145,-70,0,0],[-50,150,-85,0,0],[-60,140,-70,0,0]]
profiles.append([-52.5,150,-82.5,0,0])
pid=int(sys.argv[1]);body=dict(zip(rows[0]['body'],np.radians(profiles[pid])));out=[]
for row in rows:
 p=row['plan'];base=np.array(p['base_pose']);qs=[p['q_pre'],p['q_grasp'],p.get('q_close',p['q_grasp'])]+p['q_lift'];best=None;attempts=[]
 for dx in [0.,-.04,.04,-.08,.08]:
  solved=[];failure=None
  for index in [0,len(qs)-1]+list(range(1,len(qs)-1)):
   old=qs[index];fk=k.fk(old,base);q={**old,**body}
   for side in ['l','r']:
    target=fk[f'arm_{side}_end_link'].copy();target[0,3]+=dx
    q,pe,re=k.solve(side,target,q,base,np.eye(4),body=False)
    if pe>.004 or re>.04:failure=dict(stage=index,side=side,position_error=pe,rotation_error=re);break
   if failure:break
   payload=None if index<2 else [(f'arm_{s}_end_link',.5,[0,0,0]) for s in ['l','r']]
   check=env.check(q,payload)
   if not check['ok']:failure=dict(stage=index,envelope=check);break
   solved.append((index,q))
  attempts.append(dict(dx=dx,failure=failure))
  if failure:continue
  mapped=dict(solved);best=copy.deepcopy(p)
  for key in ['q_start','descent_path','q_descent']:best.pop(key,None)
  best.update(q_pre=mapped[0],q_grasp=mapped[1],q_close=mapped[2],q_lift=[mapped[i] for i in range(3,len(qs))],fixed_body=body,original_candidate=0)
  best['object_position'][0]+=dx;best['fixedpose_profile']=profiles[pid];best['source_baseline']=row['baseline']
  hit=collision_check(row['asset'],best)
  if hit:
   attempts[-1]['collision']=hit;best=None;continue
  break
 record=dict(asset=row['asset'],passed=best is not None,attempts=attempts,plan=best)
 out.append(record);print('SCREEN',pid,row['asset'],best is not None,attempts[-1],flush=True)
 (R/f'fixedpose_search_{pid}.json').write_text(json.dumps(out,indent=2))
print('DONE',pid,sum(r['passed'] for r in out),len(out),flush=True)
