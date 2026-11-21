from pathlib import Path
import json,numpy as np
from liftall_collision import k,robot
from audit_self_collision import intersections
R=Path(__file__).resolve().parent
ss=json.loads((R/'torquelow_budget_trials/torquelow_stoollevel_plans/stool_wood_four_leg/candidate0_repeat1/samples.json').read_text())
for s in ss:
 h=intersections(s['joint_positions'])
 if h:
  print('FIRST',s['t'],s['phase'],h,'targethits',intersections(s['joint_targets']))
  print('ARMERRORS',[(n,round(s['joint_positions'][n]-s['joint_targets'][n],5)) for n in s['joint_positions'] if 'arm_l_' in n])
  print('TARGETGAP')
  for label,q in [('actual',s['joint_positions']),('target',s['joint_targets'])]:
   f=k.fk(q);print(label,robot['body_link2'].compound_gap(robot['arm_l_link5'],np.linalg.inv(f['body_link2'])@f['arm_l_link5']))
  break
