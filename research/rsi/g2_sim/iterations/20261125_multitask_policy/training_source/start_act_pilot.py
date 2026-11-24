from pathlib import Path
import subprocess,sys,time,os
W=Path(__file__).resolve().parent
while not (W/'datasets/pilot4/manifest.json').exists():time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
subprocess.run([sys.executable,str(W/'train_g2_act.py')],cwd=W,env=env,check=True)
