from pathlib import Path
import json,numpy as np,copy,time
from g2_kinematics import Kinematics
from ideal_constraints import Envelope
from liftall_collision import check
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope();body={n:v for n,v in json.loads((R/'ground_lift_training_config.json').read_text())['fixed_joint_positions_rad'].items() if 'body_joint' in n}
plans=[];attempts=[];deadline=time.monotonic()+600
pool=[]
for folder in ['strategy15_retry_validated_plans','catalog_dual_validated_plans','strategy15_validated_plans']:
 for ci,p in enumerate(json.loads((R/folder/'bin_pedal.json').read_text())['plans']):
  if p.get('support_height_m',0)==0:pool.append((folder,ci,p))
for folder,ci,p in pool:
 base=np.array(p['base_pose']);qs=[p['q_pre'],p['q_grasp'],p.get('q_close',p['q_grasp'])]+p['q_lift']
 for dx in [0.,.02,-.02,.04,-.04,.08,-.08,.12,-.12]:
  if time.monotonic()>deadline:break
  mapped={};failure=None
  for index in [0,len(qs)-1]+list(range(1,len(qs)-1)):
   old=qs[index];fk=k.fk(old,base);q={**old,**body}
   for side in ['l','r']:
    target=fk[f'arm_{side}_end_link'].copy();target[0,3]+=dx;q,pe,re=k.solve(side,target,q,base,np.eye(4),body=False)
    if pe>.002 or re>.02:failure=dict(kind='ik',stage=index,side=side,error_m=pe);break
   if failure:break
   c=env.check(q,None if index<2 else [(f'arm_{s}_end_link',.5,np.array(p['tcp'])[:3,3]) for s in ['l','r']])
   if not c['ok']:failure=dict(kind='envelope',check=c);break
   mapped[index]=q
  if not failure:
   plan=copy.deepcopy(p)
   for key in list(plan):
    if key.startswith('descent') or key in ['q_start','q_descent']:plan.pop(key)
   plan.update(q_pre=mapped[0],q_grasp=mapped[1],q_close=mapped[2],q_lift=[mapped[i] for i in range(3,len(qs))],fixed_body=body,original_candidate=ci,source_plan_dir=folder)
   plan['object_position'][0]+=dx
   failure=check('bin_pedal',plan)
   if not failure:
    for q in [plan['q_pre'],plan['q_grasp'],plan['q_close']]+plan['q_lift']:
     hits=intersections(q)
     if hits:failure=dict(kind='visible_self',hits=hits);break
   if not failure:plans.append(plan)
  record=dict(source=folder,candidate=ci,dx=dx,failure=failure);attempts.append(record);print(record,flush=True)
  (R/'liftall_pedal_search.json').write_text(json.dumps(dict(plans=plans,attempts=attempts),indent=2))
  if not failure or len(plans)>=3:break
 if len(plans)>=3 or time.monotonic()>deadline:break
out=R/'liftall_pedal_raw_plans';out.mkdir(exist_ok=True);(out/'bin_pedal.json').write_text(json.dumps(dict(asset='bin_pedal',plans=plans),indent=2))
print('PEDAL_SEARCH_DONE',len(plans),len(attempts),flush=True)
