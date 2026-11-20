from pathlib import Path
import time,subprocess,os
R=Path(__file__).resolve().parent;pid=2748008
while Path(f'/proc/{pid}/cmdline').exists():
 try:
  if b'symhome_v4_stool_rectangular_trial.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():break
 except FileNotFoundError:break
 time.sleep(5)
with (R/'symhome_v4force_basket_repeat2.log').open('w') as f:
 r=subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/'symhome_v4force_basket_rectangular_trial.py'),'--asset','basket_rectangular','--plan-dir','symhome_plans','--repeat','2','--gpu','2'],cwd=R,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
print('BASKET_FORCE_RETRY_DONE',r.returncode,flush=True)
