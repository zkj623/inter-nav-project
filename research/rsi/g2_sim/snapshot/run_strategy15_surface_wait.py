from pathlib import Path
import subprocess,sys,time,os
R=Path(__file__).resolve().parent;aid=sys.argv[1];gpu=sys.argv[2];pd=sys.argv[3];env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
while True:
 ps=subprocess.check_output(['ps','-eo','args'],text=True)
 if not any('grasp_trial_strategy15' in l and f'--gpu {gpu}' in l and not l.startswith('bash ') for l in ps.splitlines()):break
 time.sleep(10)
subprocess.run([sys.executable,str(R/'run_strategy15_manual.py'),aid,gpu,'surface'],env=env,check=True)
