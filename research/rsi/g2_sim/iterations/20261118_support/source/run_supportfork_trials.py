from pathlib import Path
import subprocess,json,sys,time,os
R=Path(__file__).resolve().parent;aid=sys.argv[1];gpu=sys.argv[2]
py='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
deadline=time.monotonic()+1200;pd=None
while time.monotonic()<deadline:
 for variant in ('supportfork',):
  log=R/f'{variant}_{aid}.log';path=R/f'strategy15_{variant}_validated_plans'/(aid+'.json')
  if log.exists() and 'object_clear' in log.read_text(errors='replace') and path.exists() and json.loads(path.read_text())['plans']:
   pd=path.parent.name;break
 if pd:break
 time.sleep(5)
if pd is None:
 print(aid,'NO_VALIDATED_SUPPORT_PLAN',flush=True);sys.exit(0)
# Low-pose feasibility first; standing descent will be audited separately.
plans=json.loads((R/pd/(aid+'.json')).read_text())['plans']
for c in range(min(2,len(plans))):
 for repeat in (1,2,3):
  out=R/'strategy15_supportfork_trials'/pd/aid/f'candidate{c}_repeat{repeat}'
  if (out/'result.json').exists():d=json.loads((out/'result.json').read_text())
  else:
   with (R/'strategy15_logs'/f'support_{aid}_c{c}_r{repeat}.log').open('w') as log:
    subprocess.run([py,str(R/'grasp_trial_strategy15_supportfork.py'),'--asset',aid,'--plan-dir',pd,'--candidate',str(c),'--repeat',str(repeat),'--gpu',gpu],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   d=json.loads((out/'result.json').read_text())
  print({k:d.get(k) for k in ('asset','candidate','repeat','includes_descent','success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction')},flush=True)
  if not d['success']:break
 if d['success']:break
