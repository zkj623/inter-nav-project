from pathlib import Path
import subprocess,sys,time,json
W=Path(__file__).resolve().parent;O=W/'recovery_20261125'
deadline=time.monotonic()+4*3600
while not (O/'finished.json').exists():
 if (O/'stopped.json').exists():raise RuntimeError('Previous stage stopped; manual inspection required')
 if time.monotonic()>deadline:raise TimeoutError('Waiting for serial evaluations')
 time.sleep(10)
if (O/'stopped.json').exists():raise RuntimeError('Previous stage stopped')
subprocess.run([sys.executable,str(W/'resume_guarded_eval.py'),'--time-baseline'],cwd=W,check=True)
