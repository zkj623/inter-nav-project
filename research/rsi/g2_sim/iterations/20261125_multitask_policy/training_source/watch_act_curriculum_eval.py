from pathlib import Path
import subprocess,sys,time
W=Path(__file__).resolve().parent
for name,step,dataset in [('act_pilot4_refine',10000,'pilot4'),('act_pilot4_refine',30000,'pilot4'),('act_expanded12',20000,'expanded12')]:
 checkpoint=W/'runs'/name/f'step_{step:06d}.pt'
 while not checkpoint.exists():time.sleep(15)
 time.sleep(5)
 cmd=[sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(checkpoint),'--run',f'{name}_step{step:06d}','--dataset',dataset]
 if dataset=='expanded12':cmd+=['--assets','box_white','bin_lidded','basket_rectangular','bin_pedal','basket_woven','stool_rectangular']
 with (W/f'{name}_eval_{step}.log').open('w') as log:rc=subprocess.run(cmd,cwd=W,stdout=log,stderr=subprocess.STDOUT).returncode
 print('CURRICULUM_EVAL_FINISHED',name,step,rc,flush=True)
