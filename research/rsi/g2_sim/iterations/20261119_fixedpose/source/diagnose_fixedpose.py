from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in R.glob('fixedpose_*v2_trials/**/result.json'):
 d=json.loads(p.read_text());print('RESULT',p)
 print(json.dumps({k:v for k,v in d.items() if not isinstance(v,(dict,list))},indent=2))
 s=json.loads(p.with_name('samples.json').read_text())
 for phase in dict.fromkeys(x['phase'] for x in s):
  a=[x for x in s if x['phase']==phase];last=a[-1]
  print('PHASE',phase,'n',len(a),'lift',last['lift_m'],'forces',last['finger_forces'],'frozen',max(x.get('frozen_joint_max_error_rad',0) for x in a))
