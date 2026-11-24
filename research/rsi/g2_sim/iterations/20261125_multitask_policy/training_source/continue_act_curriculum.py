from pathlib import Path
import subprocess,sys,time,os
W=Path(__file__).resolve().parent
def training_active():
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   args=p.read_bytes().split(b'\0')
   if len(args)>1 and args[1].decode()==str(W/'train_g2_act.py'):return True
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return False
while not (W/'runs/act_pilot4/step_015000.pt').exists() or training_active():time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
with (W/'act_refine.log').open('w') as log:
 subprocess.run([sys.executable,str(W/'train_g2_act.py'),'--run','act_pilot4_refine','--resume',str(W/'runs/act_pilot4/step_015000.pt'),'--steps','30000','--batch','8','--lr','5e-5','--cosine','--relative-arm','--reset-action-head','--phase-balanced'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
while not (W/'datasets/expanded12/manifest.json').exists():time.sleep(15)
with (W/'act_expanded.log').open('w') as log:
 subprocess.run([sys.executable,str(W/'train_g2_act.py'),'--dataset','expanded12','--run','act_expanded12','--resume',str(W/'runs/act_pilot4_refine/step_030000.pt'),'--steps','20000','--batch','8','--lr','5e-5','--cosine','--relative-arm','--phase-balanced'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('ACT_CURRICULUM_FINISHED',flush=True)
