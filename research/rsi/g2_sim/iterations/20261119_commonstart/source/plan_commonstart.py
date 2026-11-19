from pathlib import Path
import json,copy,numpy as np,time
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,cache,check,space
from audit_self_collision import intersections
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;env=Envelope();plans={p.stem:json.loads(p.read_text())['plans'][0] for p in (R/'commonstart_base_plans').glob('*.json')}
fixed=json.loads((R/'ground_lift_training_config.json').read_text())['fixed_joint_positions_rad'];arms=[n for n in k.by_name if 'arm_' in n and k.by_name[n]['kind']=='revolute'];masters=['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1'];base=np.array(next(iter(plans.values()))['base_pose'])
def clear(a,q):
 p=plans[a];check(a,p) if a not in cache else None
 if space['collision'](q):return False
 if intersections(q):return False
 fk=k.fk(q,base);pos=np.array(p['object_position']);rot=Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix();obj=cache[a]
 for n,g in robot.items():
  t=fk[n].copy();t[:3,3]=rot.T@(t[:3,3]-pos);t[:3,:3]=rot.T@t[:3,:3]
  c=t[:3,:3]@g.bounds.mean(0)+t[:3,3];e=np.abs(t[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
  if np.any(c+e<obj.bounds[0]) or np.any(c-e>obj.bounds[1]):continue
  if obj.compound_gap(g,t)<-.001:return False
 return True
def edge(a,x,y):
 count=max(3,int(max(abs(x.get(n,0)-y.get(n,0)) for n in arms+masters)/.06)+2)
 return all(clear(a,{n:(1-u)*x.get(n,0)+u*y.get(n,0) for n in set(x)|set(y)}) for u in np.linspace(0,1,count))
homes=[]
for source in ['stool_wood_four_leg','table_side','box_white','basket_woven']:
 p=plans[source]
 for height in [.10,.20,0.]:
  q={**p['q_lift'][-1],**fixed,**{n:-.785 for n in masters}};fk=k.fk(q,base);valid=True
  for side in ['l','r']:
   t=fk[f'arm_{side}_end_link'].copy();t[2,3]+=height
   q,pe,re=k.solve(side,t,q,base,np.eye(4))
   if pe>.002 or re>.02:valid=False;break
  if valid and env.check(q)['ok']:homes.append((source,height,q))
best=None
for source,height,home in homes:
 valid=[a for a in plans if clear(a,home)]
 print('HOME',source,height,'clear',len(valid),flush=True)
 if len(valid)<len(plans):continue
 paths={}
 for a,p in plans.items():
  end=p['q_pre']
  if edge(a,home,end):paths[a]=[end]
  else:
   for lift in [.12,.25]:
    q={**end,**fixed,**{n:-.785 for n in masters}};fk=k.fk(q,base);ok=True
    for side in ['l','r']:
     t=fk[f'arm_{side}_end_link'].copy();t[2,3]+=lift;q,pe,re=k.solve(side,t,q,base,np.eye(4))
     if pe>.002 or re>.02:ok=False;break
    if ok and edge(a,home,q) and edge(a,q,end):paths[a]=[q,end];break
  print('PATH',a,a in paths,flush=True)
 if best is None or len(paths)>len(best[3]):best=(source,height,home,paths)
 if len(paths)==12:break
assert best,'No common safe home';source,height,home,paths=best
out=R/'commonstart_plans';out.mkdir(exist_ok=True)
for a,waypoints in paths.items():
 p=copy.deepcopy(plans[a]);p['q_start']=home;p['q_approach']=waypoints;p['common_start_scope']='same full q_start across all assets; fixed torso; arm/gripper approach; nominal geometry audited at <=0.06rad increments'
 (out/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[p]),indent=2))
(R/'commonstart_planning.json').write_text(json.dumps(dict(home=home,source=source,height_offset_m=height,accepted=list(paths),failed=sorted(set(plans)-set(paths))),indent=2));print('DONE',len(paths),flush=True)
