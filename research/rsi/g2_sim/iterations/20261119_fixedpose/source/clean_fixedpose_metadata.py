from pathlib import Path
import json,numpy as np
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent;env=Envelope()
for file in (R/'fixedpose_selected3_validated_plans').glob('*.json'):
 d=json.loads(file.read_text())
 for p in d['plans']:
  for key in list(p):
   if key.startswith('descent') or key in ['q_start','q_descent','q_descent_waypoints','source_plan_dir','source_candidate']:p.pop(key)
  if 'envelope_checks' in p:p['historical_source_envelope_checks']=p.pop('envelope_checks')
  p['fixedpose_envelope_checks']=[env.check(q,None if i<2 else [(f'arm_{s}_end_link',.5,np.array(p['tcp'])[:3,3]) for s in ('l','r')]) for i,q in enumerate([p['q_pre'],p['q_grasp'],p['q_close']]+p['q_lift'])]
  p['initial_pose_scope']='fixed torso/head/base, per-object pregrasp arms; no descent'
 file.write_text(json.dumps(d,indent=2))
print('Selected-plan metadata updated; executed original plans untouched')
