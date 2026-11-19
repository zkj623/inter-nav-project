from pathlib import Path
import sys,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'r1lite-collision-deps'),str(ROOT.parent/'r1lite-contact-geometry-final-20261110/research/rsi/household_catalog')]
from cooked_geometry import CookedGeometry,ConvexPart
from g2_kinematics import Kinematics
k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf')
robot={}
for r in json.loads((ROOT/'robot_assets/all_link_manifest.json').read_text()):
    d=json.loads((ROOT/'robot_assets'/(r['name']+'_cooked.json')).read_text())
    robot[r['name']]=CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(d['hulls'])])
adj={n:set() for n in k.links}
for j in k.joints:adj[j['parent']].add(j['child']);adj[j['child']].add(j['parent'])
for j in k.joints:
    if j['kind']=='fixed':
        a,b=j['parent'],j['child'];ns=(adj[a]|adj[b])-{a,b};adj[a]|=ns;adj[b]|=ns
        for n in ns:adj[n]|={a,b}
def excluded(a,b):
    near=b in adj[a] or any(b in adj[x] for x in adj[a])
    hand=any((a.startswith(f'gripper_{s}_') or a==f'arm_{s}_link7') and (b.startswith(f'gripper_{s}_') or b==f'arm_{s}_link7') for s in ('l','r'))
    head=(a.startswith('head_') or a=='body_link5') and (b.startswith('head_') or b=='body_link5')
    return near or hand or head
pairs=[(a,b) for a,b in itertools.combinations(robot,2) if not excluded(a,b)]
def collision(q):
    poses=k.fk(q);bounds={}
    for n,g in robot.items():
        t=poses[n];c=t[:3,:3]@g.bounds.mean(0)+t[:3,3];e=np.abs(t[:3,:3])@((g.bounds[1]-g.bounds[0])/2);bounds[n]=(c-e,c+e)
    for a,b in pairs:
        la,ha=bounds[a];lb,hb=bounds[b]
        if np.any(ha<lb) or np.any(hb<la):continue
        gap=robot[a].compound_gap(robot[b],np.linalg.inv(poses[a])@poses[b])
        if gap<-.001:return {'pair':[a,b],'gap_m':gap}
    return None
for aid in sys.argv[1:]:
    path=ROOT/'liftall_validated_plans'/(aid+'.json');doc=json.loads(path.read_text());accepted=[];rejected=[]
    for index,plan in enumerate(doc['plans']):
        a,b=plan['q_pre'],plan['q_grasp'];ns=set(a)|set(b)
        qs=[{n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in ns} for u in np.linspace(0,1,5)]
        qs += [plan.get('q_close',b)]+plan['q_lift']
        if 'q_start' in plan:
            start=plan['q_start'];end=plan['q_pre'];qs=[{n:(1-u)*start[n]+u*end[n] for n in end} for u in np.linspace(0,1,25)]+qs
        bad=None
        for i,q in enumerate(qs):
            hit=collision(q)
            if hit:bad=dict(candidate=index,original_candidate=plan['original_candidate'],step=i,**hit);break
        if bad:rejected.append(bad)
        else:accepted.append({**plan,'native_self_collision_checked':True})
    doc['plans']=accepted;doc['native_self_rejections']=rejected;path.write_text(json.dumps(doc,indent=2))
    print(aid,'native_clear',len(accepted),'rejected',rejected,flush=True)
