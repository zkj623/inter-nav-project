from pathlib import Path
import subprocess,sys,time
W=Path(__file__).resolve().parent
for step in [4000,12000]:
 p=W/f'runs/act_pilot4_absolute_refine/step_{step:06d}.pt'
 while not p.exists():time.sleep(15)
 with (W/f'act_absolute_refine_eval_{step}.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(p),'--run',f'act_pilot4_absolute_refine_step{step:06d}','--assets','box_white','bin_lidded','--gpu','0','--port','19029'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('ACT_ABSOLUTE_REFINE_EVAL_FINISHED',flush=True)
