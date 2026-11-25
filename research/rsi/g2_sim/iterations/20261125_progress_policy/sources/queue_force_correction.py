"""One serial successor to the already-running feedback-cadence experiment."""
from pathlib import Path
import json,os,subprocess,sys,time
W=Path(__file__).resolve().parent;O=W/'receding_act_20261125'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
priority=os.getpriority(os.PRIO_PROCESS,0)
if priority<10:os.nice(10-priority)
deadline=time.monotonic()+4*3600
while not (O/'finished.json').exists():
 if (O/'stopped.json').exists():raise RuntimeError('Previous experiment stopped; no automatic restart')
 if time.monotonic()>deadline:raise TimeoutError('Feedback experiment wait')
 time.sleep(10)
s=json.loads((W/'evaluations/act_progress_v1_execute1/summary.json').read_text());assert s['total']==4
if s['successes']<4:
 time.sleep(5)
 subprocess.run([sys.executable,str(W/'run_force_guarded.py')],cwd=W,check=True)
print('SERIAL_FORCE_CORRECTION_COMPLETE',flush=True)
