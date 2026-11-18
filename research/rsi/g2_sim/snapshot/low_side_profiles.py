from pathlib import Path
import json
from upright_constraints import grasp_body_profiles
def lowering_body_profiles():
    root=Path(__file__).resolve().parent;profiles=grasp_body_profiles()
    paths=[]
    for folder in ('lowering_trials','lowering_gravity_trials','lowering_parked_trials'):paths.extend((root/folder).glob('*/*/*/result.json'))
    for path in paths:
        d=json.loads(path.read_text());completed={s['label'] for s in d.get('completed_lowering_stages',[])}
        for stage in (d.get('plan') or {}).get('lowering_stages',[]):
            if stage['label'] not in completed:continue
            q={n:v for n,v in stage['q'].items() if 'body_joint' in n}
            if q not in profiles:profiles.append(q)
    return profiles
