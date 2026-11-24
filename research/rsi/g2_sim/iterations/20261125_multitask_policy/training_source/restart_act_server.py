from pathlib import Path
import os,signal,subprocess,sys,fcntl,socket,time
W=Path(__file__).resolve().parent;pid=675364
with (W/'gpu3_simulation.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=Path(f'/proc/{pid}/cmdline')
 if p.exists():
  assert b'act_policy_server.py' in p.read_bytes();os.kill(pid,signal.SIGTERM)
  while p.exists():time.sleep(1)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
 with (W/'act_server_v2.log').open('w') as log:server=subprocess.Popen([sys.executable,str(W/'act_policy_server.py')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
 (W/'act_server_v2.pid').write_text(str(server.pid))
 while True:
  if server.poll() is not None:raise RuntimeError('ACT server restart failed')
  try:
   with socket.create_connection(('127.0.0.1',19024),timeout=1):break
  except OSError:time.sleep(2)
print('ACT_SERVER_V2_READY',server.pid,flush=True)
