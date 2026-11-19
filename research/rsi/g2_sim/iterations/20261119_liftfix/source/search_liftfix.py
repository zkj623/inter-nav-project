from pathlib import Path
import json,numpy as np,copy,time,sys
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
from ideal_constraints import Envelope
from liftall_collision import check
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;aid=sys.argv[1];k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope()
fixed=json.loads((R/'ground_lift_training_config.json').read_text())['fixed_joint_positions_rad']
body={n:v for n,v in fixed.items() if 'body_joint' in n};deadline=time.monotonic()+720
pool=[]
for folder in ['liftall_validated_plans','strategy15_validated_plans','catalog_dual_validated_plans']:
 f=R/folder/(aid+'.json')
 if f.exists():
  for ci,p in enumerate(json.loads(f.read_text())['plans'][:6]):
   if p.get('support_height_m',0)==0:pool.append((folder,ci,p))
if aid in ['box_reinforced','bin_round_pedal']:specs=[(.03,0,0,0),(.05,0,0,0),(.07,0,0,0),(.02,.025,0,0),(.04,.04,0,0),(0,.05,0,0),(.03,.05,10,0),(.03,.05,-10,0)]
elif aid=='case_handheld':specs=[(-.02,.03,0,0),(0,.03,0,0),(.02,.03,0,0),(0,.06,0,0),(-.02,.06,0,0),(0,.03,10,0),(0,.03,-10,0)]
elif aid=='basket_rectangular':specs=[(0,.015,0,0),(0,.025,0,0),(.02,.015,0,0),(0,.015,15,0),(0,.015,-15,0),(.03,.02,20,0),(0,0,0,0)]
else:specs=[(0,0,0,.003),(0,0,0,-.003),(0,.008,0,0),(0,-.008,0,0),(0,0,10,0),(0,0,-10,0),(0,0,0,0)]
plans=[];attempts=[]
for folder,ci,p in pool:
 base=np.array(p['base_pose']);qs=[p['q_pre'],p['q_grasp'],p.get('q_close',p['q_grasp'])]+p['q_lift']
 for dx,dz,angle,centering in specs:
  if time.monotonic()>deadline:break
  if folder=='liftall_validated_plans' and (dx,dz,angle,centering)==(0,0,0,0):continue
  mapped={};failure=None
  for index in [0,len(qs)-1]+list(range(1,len(qs)-1)):
   old=qs[index];fk=k.fk(old,base);q={**old,**fixed}
   for side in ['l','r']:
    target=fk[f'arm_{side}_end_link'].copy()
    local=np.array(p['candidate'].get('contact_centers_local',{}).get(side,np.array(p['tcp'])[:3,3]))
    pivot=(target@np.r_[local,1.])[:3]
    rr=Rotation.from_euler('x',angle*(1 if side=='l' else -1),degrees=True).as_matrix()
    target[:3,:3]=rr@target[:3,:3];target[:3,3]=pivot- target[:3,:3]@local
    target[:3,3]+=[dx,0,dz]
    if side=='r':target[:3,3]+=centering*target[:3,0]
    q,pe,re=k.solve(side,target,q,base,np.eye(4),body=False)
    if pe>.002 or re>.02:failure=dict(kind='ik',stage=index,side=side,error_m=pe);break
   if failure:break
   c=env.check(q,None if index<2 else [(f'arm_{s}_end_link',.5,np.array(p['tcp'])[:3,3]) for s in ['l','r']])
   if not c['ok']:failure=dict(kind='envelope',check=c);break
   mapped[index]=q
  if not failure:
   plan=copy.deepcopy(p)
   for key in list(plan):
    if key.startswith('descent') or key in ['q_start','q_descent']:plan.pop(key)
   plan.update(q_pre=mapped[0],q_grasp=mapped[1],q_close=mapped[2],q_lift=[mapped[i] for i in range(3,len(qs))],fixed_body=body)
   plan['object_position'][0]+=dx
   plan['adaptation']=dict(source=folder,source_candidate=ci,stance_dx_m=dx,contact_dz_m=dz,wrist_x_deg=angle,right_centering_m=centering,scope='fixed original torso; physical contact validation required')
   failure=check(aid,plan)
   if not failure:
    for q in [plan['q_pre'],plan['q_grasp'],plan['q_close']]+plan['q_lift']:
     hits=intersections(q)
     if hits:failure=dict(kind='visible_self',hits=hits);break
   if not failure:plans.append(plan)
  attempts.append(dict(source=folder,candidate=ci,dx=dx,dz=dz,angle=angle,centering=centering,failure=failure));print(attempts[-1],flush=True)
  (R/f'liftfix_search_{aid}.json').write_text(json.dumps(dict(plans=plans,attempts=attempts),indent=2))
  out=R/'liftfix_plans';out.mkdir(exist_ok=True);(out/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=plans),indent=2))
  if len(plans)>=3:break
 if len(plans)>=3 or time.monotonic()>deadline:break
(R/f'liftfix_search_{aid}.done').write_text(str(len(plans)))
print('DONE',aid,len(plans),len(attempts),flush=True)
