"""Retarget verified contact paths to one torso; no object geometry changes."""
from pathlib import Path
import json,sys,copy,time
import numpy as np
from g2_kinematics import Kinematics,transform
from ideal_constraints import Envelope
from fixedpose_collision import check as collision_check
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope()
rows=[x for x in json.loads((R/'fixedpose_inventory.json').read_text()) if x['asset']=='case_handheld']
profiles=[[-50,140,-75,0,0],[-45,140,-80,0,0],[-40,140,-85,0,0],[-35,140,-90,0,0],[-30,140,-95,0,0],[-45,135,-75,0,0]]
profiles += [[-45,142,-82,0,0],[-40,142,-87,0,0],[-35,142,-92,0,0],[-35,140,-87,0,0]]
pid=int(sys.argv[1]);body=dict(zip(rows[0]['body'],np.radians(profiles[pid])));out=[]
for n,v in body.items():
 j=k.by_name[n];assert min(v-j['lower'],j['upper']-v)>=np.radians(10),n
for row in rows:
 p=row['plan'];base=np.array(p['base_pose']);qs=[p['q_pre'],p['q_grasp'],p.get('q_close',p['q_grasp'])]+p['q_lift'];best=None;attempts=[]
 for dx in sorted([float(x) for x in np.arange(-.12,.121,.01)],key=abs):
  solved=[];failure=None
  for index in [0,len(qs)-1]+list(range(1,len(qs)-1)):
   old=qs[index];fk=k.fk(old,base);q={**old,**body}
   for side in ['l','r']:
    target=fk[f'arm_{side}_end_link'].copy();target[0,3]+=dx
    q,pe,re=k.solve(side,target,q,base,np.eye(4),body=False)
    if pe>.0005 or re>.005:failure=dict(stage=index,side=side,position_error=pe,rotation_error=re);break
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
 (R/f'marginpose_fine_search_{pid}.json').write_text(json.dumps(out,indent=2))
print('DONE',pid,sum(r['passed'] for r in out),len(out),flush=True)
