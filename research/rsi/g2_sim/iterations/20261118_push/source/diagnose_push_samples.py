from pathlib import Path
import json,sys
R=Path(__file__).resolve().parent
for variant,pd in [('pushfrontstart','pushfront'),('pushreleasestart','pushrelease')]:
 p=R/f'strategy15_{variant}_trials'/f'strategy15_{pd}_validated_plans/chair_dining/candidate0_repeat1/samples.json';ss=json.loads(p.read_text())
 print(variant,'maxforward',max(s['push_forward_m'] for s in ss),'maxtracking',max(s['max_tracking_error_rad'] for s in ss))
 for s in ss[::15]+[ss[-1]]:
  print({k:s[k] for k in ['t','phase','push_forward_m','tilt_deg','total_hand_force_n','lift_m','max_tracking_error_rad']},'hands',{side:sum(f for n,f in s['finger_forces'].items() if n.startswith('gripper_'+side+'_')) for side in ['l','r']})
