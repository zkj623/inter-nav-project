from pathlib import Path
import os,subprocess,sys,time,socket,json
W=Path(__file__).resolve().parent
while 'ACT_TRAINING_COMPLETE' not in (W/'act_robust.log').read_text():time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
with (W/'act_state_skip.log').open('w') as log:subprocess.run([sys.executable,str(W/'train_g2_act.py'),'--run','act_pilot4_state_skip','--steps','8000','--batch','4','--lr','2e-5','--cosine','--relative-arm','--phase-balanced','--qpos-noise','0.02','--state-skip','--resume',str(W/'runs/act_pilot4_robust/step_012000.pt')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
while 'ACT_TRAINING_COMPLETE' not in (W/'act_absolute_refine.log').read_text():time.sleep(15)
env['CUDA_VISIBLE_DEVICES']='1'
with (W/'act_state_skip_server.log').open('w') as log:server=subprocess.Popen([sys.executable,str(W/'act_policy_server.py'),'--port','19030'],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
(W/'act_state_skip_server_pid.json').write_text(json.dumps(dict(pid=server.pid)))
while True:
 if server.poll() is not None:raise RuntimeError('State-skip ACT server exited')
 try:
  with socket.create_connection(('127.0.0.1',19030),timeout=1):break
 except OSError:time.sleep(5)
with (W/'act_state_skip_eval.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(W/'runs/act_pilot4_state_skip/step_008000.pt'),'--run','act_pilot4_state_skip_step008000','--gpu','0','--port','19030'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('ACT_STATE_SKIP_EVAL_FINISHED',flush=True)
