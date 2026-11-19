from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
s=(R/'verify_liftpair.py').read_text().replace("R.glob('liftpair*_trials/**/result.json')","list(R.glob('caseyaw_trials/**/result.json'))+list(R.glob('caseyaw_level_trials/**/result.json'))+list(R.glob('caseyaw_damped_trials/**/result.json'))").replace("'liftpair_verified_results.json'","'caseyaw_verified_results.json'")
s=s.replace("if d['success']:","if d['success'] or any(x['phase']=='hold' for x in samples):").replace("bool(not hits","bool(d['success'] and not hits")
exec(compile(s,'physical_review','exec'))
doc=json.loads((R/'commonstart_planning.json').read_text());home=doc['home'];rows=json.loads((R/'caseyaw_verified_results.json').read_text())
for r in rows:
 f=R/r['result'];d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());p=d.get('plan',{})
 r['identical_start_targets']=p.get('q_start')==home
 controlled=[n for n in home if 'arm_' in n or ('gripper' in n and k.by_name.get(n,{}).get('mimic') is None)]
 initial=next((x for x in ss if x['phase']=='settle' and x['t']>=1.5),None)
 r['initial_actual_max_error_rad']=max(abs(initial['joint_positions'].get(n,0)-home[n]) for n in controlled) if initial else None
 r['approach_recorded']=any(x['phase']=='common_approach' for x in ss)
 r['verified_success']=bool(r['verified_success'] and r['identical_start_targets'] and r['approach_recorded'] and r['initial_actual_max_error_rad'] is not None and r['initial_actual_max_error_rad']<.05)
 print('COMMON_REVIEW',r['asset'],r['verified_success'],r['initial_actual_max_error_rad'],flush=True)
(R/'caseyaw_verified_results.json').write_text(json.dumps(rows,indent=2))
