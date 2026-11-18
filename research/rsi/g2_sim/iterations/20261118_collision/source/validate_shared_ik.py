from pathlib import Path
import json,numpy as np,time
from shared_body_ik import solve_shared
from chair_core_guidance import Guidance
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent
p=json.loads((R.parent/'g2-support-20261118/strategy15_supportbranch_validated_plans/stool_rectangular.json').read_text())['plans'][0]
env=Envelope();k=env.k;base=np.array(p['base_pose']);tcp=np.array(p['tcp']);guide=Guidance('stool_rectangular')
qs=[p['q_pre'],p['q_lift'][-1]]
targets=[{s:k.fk(q,base)[f'arm_{s}_end_link']@tcp for s in ('l','r')} for q in qs]
started=time.monotonic();states,diagnostic=solve_shared(k,env,base,guide,*targets,p['q_pre'],tcp,p['object_position'],p['object_yaw_deg'])
d=dict(reference='previously verified stool_rectangular under-support path',success=states is not None,diagnostic=diagnostic,wall_seconds=time.monotonic()-started)
if states:
 d['shared_body_identical']=all(states[0].get(n)==states[1].get(n) for n in states[0] if 'body_joint' in n)
 d['envelope_checks']=[env.check(q) for q in states]
 assert d['shared_body_identical'] and all(c['ok'] for c in d['envelope_checks'])
(R/'shared_ik_validation.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
