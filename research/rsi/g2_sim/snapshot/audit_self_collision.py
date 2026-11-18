"""Exact visible-mesh self-intersection audit and baseline joint assembly pairs."""
from pathlib import Path
import sys,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'r1lite-collision-deps'))
import fcl,trimesh
from g2_kinematics import Kinematics
k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf')
names=[p.stem for p in (ROOT/'robot_assets').glob('*.npz')]
vertices={};shapes={}
for n in names:
    d=np.load(ROOT/'robot_assets'/(n+'.npz'));v=d['vertices'];faces=d['faces'];vertices[n]=v
    s=fcl.BVHModel();s.beginModel(len(v),len(faces));s.addSubModel(v,faces.astype(np.int32));s.endModel();shapes[n]=s
adj={n:set() for n in k.links}
for j in k.joints:adj[j['parent']].add(j['child']);adj[j['child']].add(j['parent'])
def near(a,b):return b in adj[a] or any(b in adj[x] for x in adj[a])
def samehand(a,b):
    return any((a.startswith(f'gripper_{s}_') or a==f'arm_{s}_link7') and (b.startswith(f'gripper_{s}_') or b==f'arm_{s}_link7') for s in ('l','r'))
def samehead(a,b):return (a.startswith('head_') or a=='body_link5') and (b.startswith('head_') or b=='body_link5')
def intersections(q,exclude=set()):
    poses=k.fk(q);bodies={};bounds={}
    for n in names:
        t=poses[n];v=vertices[n]@t[:3,:3].T+t[:3,3];bounds[n]=(v.min(0),v.max(0));bodies[n]=fcl.CollisionObject(shapes[n],fcl.Transform(t[:3,:3],t[:3,3]))
    hits=[]
    for a,b in itertools.combinations(names,2):
        key=tuple(sorted((a,b)))
        if key in exclude or near(a,b) or samehand(a,b) or samehead(a,b):continue
        la,ha=bounds[a];lb,hb=bounds[b]
        if np.any(ha<lb) or np.any(hb<la):continue
        r=fcl.CollisionResult();fcl.collide(bodies[a],bodies[b],fcl.CollisionRequest(),r)
        if r.is_collision:hits.append(key)
    return hits
if __name__=='__main__':
    out=dict(baseline_visible_intersections=intersections({}),plans={})
    for aid in sys.argv[1:]:
        d=json.loads((ROOT/'plans'/(aid+'.json')).read_text());entries=[]
        for i,p in enumerate(d['plans']):
            checks={label:intersections(q) for label,q in [('pre',p['q_pre']),('grasp',p['q_grasp']),('lift',p['q_lift'][-1])]}
            entries.append(dict(candidate=i,checks=checks))
        out['plans'][aid]=entries
    (ROOT/'self_collision_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
