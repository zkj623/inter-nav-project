from pathlib import Path
import sys,json,itertools,runpy,time,numpy as np
from ideal_constraints import Envelope
from support_collision import Checker
R=Path(__file__).resolve().parent;aid=sys.argv[1];pd=sys.argv[2];started=time.monotonic()
argv=sys.argv;sys.argv=[str(R/'strategy15_supportdesc_audit_native_self.py')]
native=runpy.run_path(sys.argv[0])['collision'];sys.argv=argv
checker=Checker(aid);env=Envelope();doc=json.loads((R/pd/(aid+'.json')).read_text());accepted=[];rejects=[]
for source_index,p in enumerate(doc['plans']):
 for dx,dy,dz in itertools.product((-.06,-.12,-.03),(.06,.12,.18),(0.,-.03,-.06)):
  if time.monotonic()-started>600:break
  park=p['q_pre'].copy();fk=checker.k.fk(park,np.array(p['base_pose']));reachable=True
  for side,sign in [('l',1),('r',-1)]:
   target=fk[f'arm_{side}_end_link'].copy();target[:3,3]+=[dx,sign*dy,dz]
   solution,pe,re=checker.k.solve(side,target,park,np.array(p['base_pose']),np.eye(4))
   if pe>.004 or re>.04:reachable=False;break
   for n,v in solution.items():
    if f'arm_{side}_joint' in n:park[n]=v
  if not reachable:rejects.append(dict(candidate=source_index,reason='parking_ik',parking=[dx,dy,dz]));continue
  start=park.copy()
  for n in start:
   if 'body_joint' in n:start[n]=0.
  bad=None;count=0
  for phase,(a,b) in enumerate(((start,park),(park,p['q_pre']))):
   for step,u in enumerate(np.linspace(0,1,25)):
    q={n:(1-u)*a[n]+u*b[n] for n in a};count+=1
    if not env.check(q)['ok']:bad={'reason':'envelope','phase':phase,'step':step};break
    hit=native(q)
    if hit:bad={'reason':'native_self','phase':phase,'step':step,**hit};break
    hit=checker.check({**p,'q_pre':q,'q_grasp':q,'q_lift':[]})
    if hit:bad={'reason':'object_clearance','phase':phase,'step':step,**hit};break
   if bad:break
  if bad:rejects.append(dict(candidate=source_index,parking=[dx,dy,dz],**bad));continue
  accepted.append({**p,'q_start':start,'descent_waypoints':[park,p['q_pre']],'descent_audit_waypoints':count,'source_plan_dir':pd,'source_candidate':source_index,'initial_pose_scope':'standing, retract-arms lowering, extend under object, lift'})
  print('CARTESIAN_DESCENT_PASS',aid,source_index,[dx,dy,dz],flush=True);break
 if accepted or time.monotonic()-started>600:break
out=R/'strategy15_supportcartesian_validated_plans';out.mkdir(exist_ok=True)
(out/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=accepted,rejections=rejects),indent=2))
print(aid,'staged descent',len(accepted),'rejected',len(rejects),flush=True)
