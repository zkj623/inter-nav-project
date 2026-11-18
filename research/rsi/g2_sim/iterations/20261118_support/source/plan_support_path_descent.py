from pathlib import Path
import sys,json,itertools,runpy,time,numpy as np
from ideal_constraints import Envelope
from support_state_collision import Checker
R=Path(__file__).resolve().parent;aid=sys.argv[1];pd=sys.argv[2];started=time.monotonic()
argv=sys.argv;sys.argv=[str(R/'strategy15_supportdesc_audit_native_self.py')]
native=runpy.run_path(sys.argv[0])['collision'];sys.argv=argv
checker=Checker(aid);env=Envelope();doc=json.loads((R/pd/(aid+'.json')).read_text());out=[];reject=[]
for ci,p in enumerate(doc['plans']):
 def check(q):
  if not env.check(q)['ok']:return 'envelope'
  hit=native(q)
  if hit:return 'self:'+str(hit)
  hit=checker.check({**p,'q_pre':q,'q_grasp':q,'q_lift':[]})
  return 'object:'+str(hit) if hit else None
 for dx,dy,dz in itertools.product((-.06,-.12,-.18),(.05,.09,.13),(.20,.30)):
  if time.monotonic()-started>600:break
  q=p['q_pre'].copy();fk=checker.k.fk(q,np.array(p['base_pose']));initial={s:fk[f'arm_{s}_end_link'].copy() for s in ('l','r')};path=[q.copy()];bad=None
  for phase in (0,1):
   for u in np.linspace(.1,1.,10):
    for side,sign in [('l',1),('r',-1)]:
     target=initial[side].copy()
     target[:3,3]+=np.array([dx,sign*dy,0])*(u if phase==0 else 1.)
     if phase==1:target[2,3]+=dz*u
     sol,pe,re=checker.k.solve(side,target,q,np.array(p['base_pose']),np.eye(4))
     if pe>.004 or re>.04:bad='arm_ik';break
     for n,v in sol.items():
      if f'arm_{side}_joint' in n:q[n]=v
    if bad:break
    bad=check(q)
    if bad:break
    path.append(q.copy())
   if bad:break
  if not bad:
   start=q.copy()
   for n in start:
    if 'body_joint' in n:start[n]=0.
   for u in np.linspace(0,1,25):
    state={n:(1-u)*start[n]+u*q[n] for n in q};bad=check(state)
    if bad:break
  if bad:reject.append(dict(candidate=ci,offset=[dx,dy,dz],reason=bad));continue
  waypoints=list(reversed(path))
  out.append({**p,'q_start':start,'descent_waypoints':waypoints,'descent_durations_seconds':[12.]+[.4]*(len(waypoints)-1),'source_plan_dir':pd,'source_candidate':ci,'initial_pose_scope':'standing then hands outward, lowering, under-seat insertion','descent_checked_states':len(path)+25})
  print('PATH_DESCENT_PASS',aid,ci,[dx,dy,dz],flush=True);break
 if out or time.monotonic()-started>600:break
dest=R/'strategy15_supportpath_validated_plans';dest.mkdir(exist_ok=True);(dest/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=out,rejections=reject),indent=2))
print(aid,'path descent',len(out),'rejected',len(reject),flush=True)
