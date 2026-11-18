from pathlib import Path
import json,time,subprocess,os,sys
R=Path(__file__).resolve().parent;aid,gpu=sys.argv[1:];pd='strategy15_pushmid_validated_plans'
for _ in range(240):
 log=R/f'pushmid_{aid}.log'
 if log.exists() and 'PUSHMID_AUDITS_COMPLETE' in log.read_text():break
 time.sleep(5)
else:raise RuntimeError('Planning did not complete')
plans=json.loads((R/pd/(aid+'.json')).read_text())['plans'];print(aid,'plans',len(plans),flush=True)
for c in range(min(3,len(plans))):
 with (R/f'pushmid_trial_{aid}_{c}.log').open('w') as log:
  subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/'pushmidstart_trial_strategy15.py'),'--asset',aid,'--plan-dir',pd,'--candidate',str(c),'--repeat','1','--gpu',gpu],cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((R/'strategy15_pushmidstart_trials'/pd/aid/f'candidate{c}_repeat1/result.json').read_text());print({k:d.get(k) for k in ['asset','candidate','success','failure','push_forward_m','max_push_tilt_deg','max_hand_force_n']},flush=True)
 if d['success']:break
