from pathlib import Path
import sys,subprocess,json,os
R=Path(__file__).resolve().parent;variant,pd,aid,c,gpu=sys.argv[1:]
py='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
for rep in (2,3):
 out=R/f'strategy15_{variant}_trials'/pd/aid/f'candidate{c}_repeat{rep}'
 if not (out/'result.json').exists():
  with (R/'strategy15_logs'/f'{variant}_{aid}_repeat{rep}.log').open('w') as log:
   subprocess.run([py,str(R/f'grasp_trial_strategy15_{variant}.py'),'--asset',aid,'--plan-dir',pd,'--candidate',c,'--repeat',str(rep),'--gpu',gpu],cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((out/'result.json').read_text());print({k:d.get(k) for k in ('asset','repeat','success','failure','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction')},flush=True)
 if not d['success']:break
