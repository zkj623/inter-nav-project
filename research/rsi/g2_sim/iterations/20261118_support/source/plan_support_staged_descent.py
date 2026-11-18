from pathlib import Path
import sys,json,itertools,runpy,time,numpy as np
from ideal_constraints import Envelope
from support_collision import Checker
R=Path(__file__).resolve().parent;aid=sys.argv[1];pd=sys.argv[2];started=time.monotonic()
argv=sys.argv;sys.argv=[str(R/'strategy15_supportdesc_audit_native_self.py')]
native=runpy.run_path(sys.argv[0])['collision'];sys.argv=argv
checker=Checker(aid);env=Envelope();doc=json.loads((R/pd/(aid+'.json')).read_text());accepted=[];rejects=[]
for source_index,p in enumerate(doc['plans']):
 for a1,a2,a4 in itertools.product((1.4,1.,.6),(-.6,-1.),(-1.8,-1.2)):
  if time.monotonic()-started>600:break
  park=p['q_pre'].copy()
  for side,sign,prefix in [('l',1,2),('r',-1,6)]:
   park[f'idx{prefix}1_arm_{side}_joint1']=sign*a1
   park[f'idx{prefix}2_arm_{side}_joint2']=a2
   park[f'idx{prefix}4_arm_{side}_joint4']=a4
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
  if bad:rejects.append(dict(candidate=source_index,parking=[a1,a2,a4],**bad));continue
  accepted.append({**p,'q_start':start,'descent_waypoints':[park,p['q_pre']],'descent_audit_waypoints':count,'source_plan_dir':pd,'source_candidate':source_index,'initial_pose_scope':'standing, retract-arms lowering, extend under object, lift'})
  print('STAGED_DESCENT_PASS',aid,source_index,[a1,a2,a4],flush=True);break
 if accepted or time.monotonic()-started>600:break
out=R/'strategy15_supportstaged_validated_plans';out.mkdir(exist_ok=True)
(out/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=accepted,rejections=rejects),indent=2))
print(aid,'staged descent',len(accepted),'rejected',len(rejects),flush=True)
