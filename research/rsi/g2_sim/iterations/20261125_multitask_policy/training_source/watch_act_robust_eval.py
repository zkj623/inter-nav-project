from pathlib import Path
import json,subprocess,sys,time
W=Path(__file__).resolve().parent
while 'SECONDARY_ACT_EVAL_FINISHED' not in (W/'act_secondary_supervisor.log').read_text():time.sleep(15)
for step in [4000,12000]:
 p=W/f'runs/act_pilot4_robust/step_{step:06d}.pt'
 while not p.exists():time.sleep(15)
 with (W/f'act_robust_eval_{step}.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(p),'--run',f'act_pilot4_robust_step{step:06d}','--assets','box_white','bin_lidded','bin_pedal','basket_rectangular','--gpu','0','--port','19029'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('ACT_ROBUST_EVAL_FINISHED',flush=True)
