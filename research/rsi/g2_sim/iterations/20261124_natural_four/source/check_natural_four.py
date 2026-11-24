from pathlib import Path
import json,numpy as np,sys
R=Path(__file__).resolve().parent;folder=sys.argv[1] if len(sys.argv)>1 else 'natural_v1'
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'geometry_lib','exec'))
rows=[]
for f in (R/folder).glob('*.json'):
 d=json.loads(f.read_text());p=d['plans'][0];a=d['asset'];plans[a]=p
 approach=[p['q_start']]+p.get('q_approach',[])+[p['q_pre']]
 contact=[p['q_pre']]+p.get('q_contact_approach',[p['q_grasp']])
 check_plan=dict(p)
 if 'q_contact_approach' in p:check_plan['q_pre']=p['q_grasp']
 row={'asset':a,'approach':all(edge(a,x,y) for x,y in zip(approach,approach[1:])),'contact_approach':all(edge(a,x,y) for x,y in zip(contact,contact[1:])),'grasp_lift':check(a,check_plan)};rows.append(row);print(row,flush=True)
(R/(folder+'_geometry.json')).write_text(json.dumps(rows,indent=2))
