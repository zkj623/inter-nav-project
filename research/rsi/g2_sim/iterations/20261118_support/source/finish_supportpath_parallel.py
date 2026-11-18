from pathlib import Path
import time,json,subprocess,os
R=Path(__file__).resolve().parent
pd='strategy15_supportpath_validated_plans';aid='stool_rectangular'
root=R/'strategy15_supportpath_trials'/pd/aid
first=root/'candidate0_repeat1/result.json'
for _ in range(180):
 if first.exists():break
 time.sleep(10)
else:raise RuntimeError('First full-path trial did not complete')
d=json.loads(first.read_text());print({k:d.get(k) for k in ['success','failure','includes_descent','min_hold_lift_m','max_hold_tilt_deg']},flush=True)
if not d['success']:raise SystemExit(0)
jobs=[]
for repeat,gpu in [(2,5),(3,6)]:
 if (root/f'candidate0_repeat{repeat}/result.json').exists():continue
 log=(R/f'trial_supportpath_repeat{repeat}.log').open('w')
 cmd=['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/'grasp_trial_strategy15_supportpath.py'),'--asset',aid,'--plan-dir',pd,'--candidate','0','--repeat',str(repeat),'--gpu',str(gpu)]
 jobs.append((repeat,subprocess.Popen(cmd,cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT),log))
for repeat,proc,log in jobs:
 rc=proc.wait();log.close()
 if rc:raise RuntimeError((repeat,rc))
 d=json.loads((root/f'candidate0_repeat{repeat}/result.json').read_text())
 print({k:d.get(k) for k in ['repeat','success','failure','includes_descent','min_hold_lift_m','max_hold_tilt_deg']},flush=True)
