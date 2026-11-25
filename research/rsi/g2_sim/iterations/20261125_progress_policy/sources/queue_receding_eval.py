from pathlib import Path
import json,time,subprocess,sys
W=Path(__file__).resolve().parent;O=W/'progress_act_20261125';deadline=time.monotonic()+4*3600
while not (O/'time_baseline_finished.json').exists():
 if (O/'stopped.json').exists():raise RuntimeError('Previous experiment stopped')
 if time.monotonic()>deadline:raise TimeoutError('Evaluation wait')
 time.sleep(10)
s=json.loads((W/'evaluations/act_progress_v1_8000/summary.json').read_text())
if s['successes']<4:
 time.sleep(5);subprocess.run([sys.executable,str(W/'run_receding_guarded.py')],cwd=W,check=True)
 s1=json.loads((W/'evaluations/act_progress_v1_execute1/summary.json').read_text())
 if max(s['successes'],s1['successes'])<2:
  time.sleep(5);subprocess.run([sys.executable,str(W/'run_timeonly_guarded.py')],cwd=W,check=True)
print('SERIAL_CORRECTION_QUEUE_COMPLETE',flush=True)
