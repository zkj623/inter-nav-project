from pathlib import Path
import subprocess,sys,time
W=Path(__file__).resolve().parent;p=W/'runs/act_pilot4_refine/step_003000.pt'
while not p.exists():time.sleep(15)
with (W/'act_relative_early_eval.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(p),'--run','act_pilot4_refine_step003000','--assets','box_white','bin_lidded'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('RELATIVE_EARLY_EVAL_FINISHED',flush=True)
