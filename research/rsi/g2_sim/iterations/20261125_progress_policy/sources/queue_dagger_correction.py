from pathlib import Path
import os,json,subprocess,time,sys
W=Path(__file__).resolve().parent;O=W/'force_act_20261125';deadline=time.monotonic()+4*3600
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);priority=os.getpriority(os.PRIO_PROCESS,0)
if priority<10:os.nice(10-priority)
while not (O/'time_baseline_finished.json').exists():
 if (O/'stopped.json').exists():raise RuntimeError('Force experiment stopped; no automatic retry')
 if time.monotonic()>deadline:raise TimeoutError('Force experiment wait')
 time.sleep(10)
result=json.loads((W/'evaluations/act_force_v1_8000/summary.json').read_text());assert result['total']==4
if result['successes']<2:
 time.sleep(5);subprocess.run([sys.executable,str(W/'run_dagger_guarded.py')],cwd=W,check=True)
print('SERIAL_DAGGER_CORRECTION_QUEUE_COMPLETE',flush=True)
