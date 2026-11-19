from pathlib import Path
import json,time,numpy as np
R=Path(__file__).resolve().parent
while not (R/'commonstart_planning.json').exists():time.sleep(5)
code=(R/'plan_commonstart.py').read_text().split('homes=[]')[0];exec(compile(code,'commonstart_library','exec'))
doc=json.loads((R/'commonstart_planning.json').read_text());home=doc['home'];results=[]
for a in doc['failed']:
 p=plans[a];end=p['q_pre'];path=None
 # Stagger arms: each waypoint still uses one fixed common initial state.
 for first in ['l','r']:
  q=home.copy();q.update({n:v for n,v in end.items() if f'arm_{first}_' in n or f'gripper_{first}_' in n})
  if edge(a,home,q) and edge(a,q,end):path=[q,end];break
 if path is None:
  rng=np.random.default_rng(42);nodes=[home];parents=[-1];deadline=time.monotonic()+600
  for attempt in range(600):
   if time.monotonic()>deadline:break
   target=end.copy() if attempt%4==0 else {**home,**{n:float(np.clip((home.get(n,0)+end.get(n,0))/2+rng.normal(0,.65),k.by_name[n]['lower']+.01,k.by_name[n]['upper']-.01)) for n in arms}}
   nearest=min(range(len(nodes)),key=lambda i:sum((nodes[i].get(n,0)-target.get(n,0))**2 for n in arms));near=nodes[nearest];dist=max(abs(near.get(n,0)-target.get(n,0)) for n in arms);u=min(1.,.3/max(.001,dist));q={n:(1-u)*near.get(n,0)+u*target.get(n,0) for n in set(near)|set(target)}
   if not edge(a,near,q):continue
   nodes.append(q);parents.append(nearest)
   if edge(a,q,end):
    chain=[end];i=len(nodes)-1
    while i>0:chain.append(nodes[i]);i=parents[i]
    path=list(reversed(chain));break
 if path:
  out=json.loads((R/'commonstart_base_plans'/(a+'.json')).read_text());p=out['plans'][0];p.update(q_start=home,q_approach=path,common_start_scope='same q_start; staged/RRT joint path audited <=0.06rad increments');(R/'commonstart_extra_plans').mkdir(exist_ok=True);(R/'commonstart_extra_plans'/(a+'.json')).write_text(json.dumps(out,indent=2))
 results.append(dict(asset=a,path=path));print('EXTRA_PATH',a,'found',path is not None,flush=True)
(R/'commonstart_extra_planning.json').write_text(json.dumps(results,indent=2))
