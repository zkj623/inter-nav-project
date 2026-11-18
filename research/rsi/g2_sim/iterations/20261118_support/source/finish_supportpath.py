from pathlib import Path
import time,json,subprocess
R=Path(__file__).resolve().parent
p=R/'strategy15_supportpath_trials/strategy15_supportpath_validated_plans/stool_rectangular/candidate0_repeat1/result.json'
for _ in range(180):
 if p.exists():break
 time.sleep(10)
else:raise RuntimeError('First full-path trial did not complete')
d=json.loads(p.read_text());print({k:d.get(k) for k in ['success','failure','includes_descent','min_hold_lift_m','max_hold_tilt_deg']},flush=True)
if d['success']:subprocess.run(['python3',str(R/'run_support_repeats.py'),'supportpath','strategy15_supportpath_validated_plans','stool_rectangular','0','5'],cwd=R,check=True)
