from pathlib import Path
import json,copy,numpy as np
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
search=json.loads((R/'symhome_search.json').read_text());home=search['candidates'][0]['q'];default=search['default'];initway=search['candidates'][0]['waypoints'];cfg=json.loads((R/'ground_lift_training_config.json').read_text())
out=R/'symhome_plans';out.mkdir(exist_ok=True);records=[]
for a,b in cfg['validated_simulation_baselines'].items():
 doc=json.loads((R/b['plan_dir']/(a+'.json')).read_text());p=copy.deepcopy(doc['plans'][b['candidate']]);plans[a]=p
 initok=all(edge(a,x,y) for x,y in zip([default]+initway,initway));way=None
 if initok and edge(a,home,p['q_pre']):way=[p['q_pre']]
 if initok and way is None:
  for h in [.12,.20,.30]:
   q=p['q_pre'].copy();fk=k.fk(q,base);valid=True
   for side in ['l','r']:
    t=fk[f'arm_{side}_end_link'].copy();t[2,3]+=h;q,pe,re=k.solve(side,t,q,base,np.eye(4))
    if pe>.002 or re>.02:valid=False;break
   if valid and edge(a,home,q) and edge(a,q,p['q_pre']):way=[q,p['q_pre']];break
 p.update(q_start=home,q_approach=way or [],q_default=default,q_prepare=initway)
 (out/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[p] if way else []),indent=2))
 row=dict(asset=a,preparation_clear=initok,approach_clear=bool(way),waypoints=len(way or []),old_baseline=b);records.append(row);print(row,flush=True)
(R/'symhome_connection.json').write_text(json.dumps(records,indent=2));print('DONE',sum(r['approach_clear'] for r in records),flush=True)
