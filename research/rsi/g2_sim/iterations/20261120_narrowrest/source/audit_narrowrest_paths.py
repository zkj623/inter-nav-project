from pathlib import Path
import json
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
records=[]
for f in sorted((R/'narrowrest_plans').glob('*.json')):
 d=json.loads(f.read_text());a=f.stem;p=d['plans'][0];plans[a]=p
 qs=[p['q_default']]+p['q_prepare']+p['q_approach'];ok=all(edge(a,x,y) for x,y in zip(qs,qs[1:]))
 records.append(dict(asset=a,standby_preparation_approach_clear=ok));print(a,ok,flush=True)
(R/'narrowrest_path_audit.json').write_text(json.dumps(records,indent=2));print('PATHS_DONE',sum(r['standby_preparation_approach_clear'] for r in records),flush=True)
