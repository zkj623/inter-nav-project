from pathlib import Path
import json,copy,numpy as np
from g2_kinematics import Kinematics
from ideal_constraints import Envelope
from audit_self_collision import intersections
from liftfix_arc_collision import check
R=Path(__file__).resolve().parent;k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');env=Envelope();records=[]
for aid in ['case_handheld','box_reinforced']:
 original=json.loads((R/'liftfix_pedal_plans'/(aid+'.json')).read_text())['plans'];out=[]
 for ci,old in enumerate(original):
  p=copy.deepcopy(old);base=np.array(p['base_pose']);lift=[];failure=None
  for i,q in enumerate(p['q_lift']):
   fk=k.fk(q,base);updated=q.copy()
   for side in ['l','r']:
    t=fk[f'arm_{side}_end_link'].copy();t[0,3]+=.06*min(1.,.025*(i+1)/.1)
    updated,pe,re=k.solve(side,t,updated,base,np.eye(4))
    if pe>.002 or re>.02:failure=dict(kind='ik',stage=i,side=side,error_m=pe);break
   if failure:break
   if not env.check(updated)['ok']:failure=dict(kind='envelope',stage=i);break
   if intersections(updated):failure=dict(kind='visible_self',stage=i);break
   lift.append(updated)
  if not failure:
   p['q_lift']=lift;p['planned_lift_forward_m']=.06;p['lift_path_note']='6cm forward wrist motion during first 10cm lift; no object pose control';p['geometry_source_candidate']=ci
   failure=check(aid,p)
   if not failure:out.append(p)
  records.append(dict(asset=aid,source_candidate=ci,contact=p['candidate'],failure=failure));print(aid,ci,p['candidate']['tilt_deg'],p['candidate']['height_fraction'],p['object_position'],failure,flush=True)
 folder=R/'liftpair_geometry_plans';folder.mkdir(exist_ok=True);(folder/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=out),indent=2))
(R/'liftpair_geometry_search.json').write_text(json.dumps(records,indent=2))
