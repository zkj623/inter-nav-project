from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
for p in R.glob('marginpose_*_trials/**/result.json'):
 d=json.loads(p.read_text());print('RESULT',str(p.relative_to(R)),{k:d.get(k) for k in ['success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','support_balance_deg']})
 ss=json.loads(p.with_name('samples.json').read_text())
 for phase in dict.fromkeys(s['phase'] for s in ss):
  a=[s for s in ss if s['phase']==phase];s=a[-1];forces=s['finger_forces'];f={side:[sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_{finger}_')) for finger in ['inner','outer']] for side in ['l','r']}
  v=float(np.linalg.norm(np.array(a[-1]['object_position'])-a[-2]['object_position'])/(a[-1]['t']-a[-2]['t'])) if len(a)>1 else None
  print(phase,len(a),'lift',s['lift_m'],'tilt',s['tilt_deg'],'forces',f,'velocity',v)
