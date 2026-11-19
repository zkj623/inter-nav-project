from pathlib import Path
import json,copy,numpy as np
from g2_kinematics import Kinematics
from ideal_constraints import Envelope
from audit_self_collision import intersections
R=Path(__file__).resolve().parent
s=(R/'liftall_collision.py').read_text().replace("rot=Rotation.from_euler('z'", "pos[0]+=p.get('planned_lift_forward_m',0.)*min(1.,dz/.1);rot=Rotation.from_euler('z'")
(R/'liftfix_arc_collision.py').write_text(s)
from liftfix_arc_collision import check
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope();records=[]
for aid in ['case_handheld','bin_round_pedal']:
 original=json.loads((R/'liftfix_plans'/(aid+'.json')).read_text())['plans'][1];out=[]
 for forward in [.04,.06,.08]:
  p=copy.deepcopy(original);base=np.array(p['base_pose']);lift=[];failure=None
  for i,q in enumerate(p['q_lift']):
   fk=k.fk(q,base);updated=q.copy()
   for side in ['l','r']:
    t=fk[f'arm_{side}_end_link'].copy();t[0,3]+=forward*min(1.,.025*(i+1)/.1)
    updated,pe,re=k.solve(side,t,updated,base,np.eye(4))
    if pe>.002 or re>.02:failure=dict(kind='ik',stage=i,side=side,error_m=pe);break
   if failure:break
   if not env.check(updated)['ok']:failure=dict(kind='envelope',stage=i);break
   if intersections(updated):failure=dict(kind='visible_self',stage=i);break
   lift.append(updated)
  if not failure:
   p['q_lift']=lift;p['planned_lift_forward_m']=forward;p['lift_path_note']='Move wrists forward gradually during first 10cm of lift, then vertical; expected object path used only for collision audit; object pose is never driven.'
   failure=check(aid,p)
   if not failure:out.append(p)
  records.append(dict(asset=aid,forward_m=forward,failure=failure));print(records[-1],flush=True)
 folder=R/'liftfix_arc_plans';folder.mkdir(exist_ok=True);(folder/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=out),indent=2))
(R/'liftfix_arc_search.json').write_text(json.dumps(records,indent=2))
