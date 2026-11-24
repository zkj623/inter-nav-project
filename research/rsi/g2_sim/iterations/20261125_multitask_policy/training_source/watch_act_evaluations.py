from pathlib import Path
import subprocess,sys,time,json
W=Path(__file__).resolve().parent;deadline=time.monotonic()+7*3600
for step in [0,1000,3000,8000,15000]:
 ckpt=W/'runs/act_pilot4'/f'step_{step:06d}.pt'
 while not ckpt.exists():
  if time.monotonic()>deadline:raise TimeoutError('ACT checkpoint deadline '+str(step))
  time.sleep(15)
 # Checkpoint writer uses copy2 for retained snapshots; wait for a stable complete file.
 time.sleep(5)
 cmd=[sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(ckpt),'--run',f'act_pilot4_step{step:06d}']
 if step==0:cmd+=['--assets','box_white','bin_lidded']
 with (W/f'act_eval_step{step:06d}.log').open('w') as log:rc=subprocess.run(cmd,cwd=W,stdout=log,stderr=subprocess.STDOUT).returncode
 print('ACT_EVAL_FINISHED',step,rc,flush=True)
