from pathlib import Path
import time,json,subprocess,sys
W=Path(__file__).resolve().parent;O=W/'progress_act_20261125';deadline=time.monotonic()+4*3600
while not (O/'time_baseline_finished.json').exists():
 if (O/'stopped.json').exists():raise RuntimeError('Previous run stopped; inspect before proceeding')
 if time.monotonic()>deadline:raise TimeoutError('Waiting for progress ACT evaluation')
 time.sleep(10)
s=json.loads((W/'evaluations/act_progress_v1_8000/summary.json').read_text())
if s['successes']>=2:
 print('SKIP_TIME_ONLY: progress policy already passed multiple objects',flush=True)
else:
 time.sleep(5)
 subprocess.run([sys.executable,str(W/'run_timeonly_guarded.py')],cwd=W,check=True)
