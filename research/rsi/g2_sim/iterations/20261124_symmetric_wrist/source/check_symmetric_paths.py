from pathlib import Path
import json,sys,numpy as np,concurrent.futures
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
prep=json.loads((R/'standby_preparation_config.json').read_text());rows=[]
for f in sorted((R/'realpose_plans').glob('*.json')):
 d=json.loads(f.read_text())
 if 'plans' not in d:continue
 a=d['asset'];p=d['plans'][0];plans[a]=p
 stages={'standby_to_home':[prep['standby_joint_positions_rad'],p['q_start']],'home_to_pregrasp':[p['q_start']]+p['q_approach']+[p['q_pre']]}
 row=dict(asset=a)
 for phase,qs in stages.items():row[phase]=all(edge(a,x,y) for x,y in zip(qs,qs[1:]))
 row['grasp_lift_geometry']=check(a,p) is None;rows.append(row);print(row,flush=True)
(R/'symmetric_paths.json').write_text(json.dumps(rows,indent=2));print('PATH_AUDIT_COMPLETE',flush=True)
