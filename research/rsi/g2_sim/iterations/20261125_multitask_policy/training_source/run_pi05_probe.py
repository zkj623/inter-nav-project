from pathlib import Path
import subprocess,sys,os,time,socket
W=Path(__file__).resolve().parent;checkpoint=W/'checkpoints/g2_pi05_lora/pilot4/1000'
while not (checkpoint/'params').exists():time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',OPENPI_DATA_HOME=str(W/'openpi_cache'))
with (W/'pi05_probe_server.log').open('w') as log:server=subprocess.Popen([str(W/'vendor/openpi/.venv/bin/python'),str(W/'pi05_policy_server.py'),'--port','19026'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
try:
 while True:
  if server.poll() is not None:raise RuntimeError('pi05 probe server exited')
  try:
   with socket.create_connection(('127.0.0.1',19026),timeout=1):break
  except OSError:time.sleep(5)
 subprocess.run([sys.executable,str(W/'probe_pi05_checkpoint.py'),'--checkpoint',str(checkpoint)],cwd=W,check=True)
finally:
 server.terminate();server.wait(timeout=60)
