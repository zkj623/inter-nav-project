from pathlib import Path
import subprocess,sys,os,time,socket,json
W=Path(__file__).resolve().parent;p=W/'runs/act_pilot4_refine/step_003000.pt'
while True:
 metrics=W/'runs/act_pilot4_refine/metrics.jsonl'
 if p.exists() and metrics.exists() and any(json.loads(line).get('step',0)>3000 for line in metrics.read_text().splitlines()):break
 time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
with (W/'act_secondary_server.log').open('w') as log:server=subprocess.Popen([sys.executable,str(W/'act_policy_server.py'),'--port','19029'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
(W/'act_secondary_server_pid.json').write_text(json.dumps(dict(pid=server.pid)))
while True:
 if server.poll() is not None:raise RuntimeError('Secondary ACT server exited')
 try:
  with socket.create_connection(('127.0.0.1',19029),timeout=1):break
 except OSError:time.sleep(5)
# Independent simulation slot and inference instance; no checkpoint switching in other rollouts.
with (W/'act_secondary_eval.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(p),'--run','act_pilot4_refine_step003000_execute50','--execute','50','--assets','box_white','bin_lidded','--gpu','0','--port','19029'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('SECONDARY_ACT_EVAL_FINISHED',flush=True)
