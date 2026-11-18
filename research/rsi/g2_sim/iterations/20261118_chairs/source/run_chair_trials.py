from pathlib import Path
import time,subprocess,json,sys,os
R=Path(__file__).resolve().parent;aid,gpu=sys.argv[1:]
pending={'chaircorners','chairlegs'};deadline=time.monotonic()+1800
while pending and time.monotonic()<deadline:
 for variant in sorted(pending.copy()):
  log=R/f'{variant}_{aid}.log';pd=f'strategy15_{variant}_validated_plans';p=R/pd/(aid+'.json')
  if not log.exists() or 'object_clear' not in log.read_text() or not p.exists():continue
  plans=json.loads(p.read_text())['plans'];pending.remove(variant)
  print(aid,variant,'validated plans',len(plans),flush=True)
  for c in range(min(2,len(plans))):
   with (R/f'trial_{variant}_{aid}_{c}.log').open('w') as out:
    subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/f'grasp_trial_strategy15_{variant}.py'),'--asset',aid,'--plan-dir',pd,'--candidate',str(c),'--repeat','1','--gpu',gpu],cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=out,stderr=subprocess.STDOUT,check=True)
   d=json.loads((R/f'strategy15_{variant}_trials'/pd/aid/f'candidate{c}_repeat1/result.json').read_text())
   print({k:d.get(k) for k in ['asset','candidate','success','failure','includes_descent','min_hold_lift_m','max_hold_tilt_deg']},flush=True)
   if d['success']:break
 if pending:time.sleep(5)
if pending:raise RuntimeError('Planners incomplete: '+str(pending))
