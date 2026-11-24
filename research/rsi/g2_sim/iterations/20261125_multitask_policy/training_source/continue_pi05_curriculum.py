from pathlib import Path
import subprocess,sys,os,json,time,signal,socket
W=Path(__file__).resolve().parent
# Train the relative variant after baseline training; baseline evaluation uses its own server.
while Path('/proc/'+str(json.loads((W/'pi05_train_pid.json').read_text())['pid'])).exists():time.sleep(30)
base=W/'checkpoints/g2_pi05_lora/pilot4';checkpoint=max([p for p in base.iterdir() if p.name.isdigit() and (p/'params').exists()],key=lambda p:int(p.name))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',OPENPI_DATA_HOME=str(W/'openpi_cache'),WANDB_MODE='disabled')
with (W/'pi05_relative_train.log').open('w') as log:
 subprocess.run([str(W/'vendor/openpi/.venv/bin/python'),str(W/'train_g2_pi05_relative.py'),'--run','pilot4_relative','--steps','5000','--batch','2','--init-checkpoint',str(checkpoint/'params')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
base=W/'checkpoints/g2_pi05_lora/pilot4_relative';checkpoint=max([p for p in base.iterdir() if p.name.isdigit() and (p/'params').exists()],key=lambda p:int(p.name))
with (W/'pi05_relative_server.log').open('w') as log:server=subprocess.Popen([str(W/'vendor/openpi/.venv/bin/python'),str(W/'pi05_policy_server.py'),'--port','19028','--scope','pi05_pilot4_relative_scope.json'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
(W/'pi05_relative_server_pid.json').write_text(json.dumps(dict(pid=server.pid)))
while True:
 if server.poll() is not None:raise RuntimeError('relative pi05 server exited')
 try:
  with socket.create_connection(('127.0.0.1',19028),timeout=1):break
 except OSError:time.sleep(5)
with (W/'pi05_relative_eval.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(checkpoint),'--run','pi05_pilot4_relative_step'+checkpoint.name,'--port','19028'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PI05_RELATIVE_CURRICULUM_FINISHED',flush=True)

