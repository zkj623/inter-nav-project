from pathlib import Path
import os, sys, json, time, subprocess, signal, shutil, socket, fcntl
W=Path(__file__).resolve().parent
PY='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
OUT=W/'progress_act_20261125'; OUT.mkdir(exist_ok=True)
lock=(W/'recovery_20261125/supervisor.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
children=[]; previous=None; strikes=0
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
priority=os.getpriority(os.PRIO_PROCESS,0)
if priority<10:os.nice(10-priority)
def snapshot():
 global previous
 mem={a:int(b.split()[0]) for a,b in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
 vm=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
 cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
 now=time.monotonic(); swap=int(vm['pswpin'])+int(vm['pswpout']); rate=wait=0
 if previous:
  dt=now-previous[0]; rate=(swap-previous[1])*os.sysconf('SC_PAGE_SIZE')/dt/2**20
  delta=sum(cpu)-sum(previous[2]); wait=100*(cpu[4]-previous[2][4])/max(delta,1)
 previous=(now,swap,cpu)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 gpus=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()]
 s=dict(time=time.strftime('%FT%T'),available_gib=mem['MemAvailable']/2**20,swap_used_gib=(mem['SwapTotal']-mem['SwapFree'])/2**20,swap_io_mib_s=rate,iowait_percent=wait,load=os.getloadavg(),free_root_gib=shutil.disk_usage('/').free/2**30,free_data_gib=shutil.disk_usage(W).free/2**30,gpus=gpus)
 with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
 return s
def check():
 global strikes
 s=snapshot(); bad=s['available_gib']<32 or s['swap_io_mib_s']>20 or s['iowait_percent']>25 or s['free_root_gib']<30 or s['free_data_gib']<100 or any(g[2]>85 or (g[0] in (0,1) and g[1]<2048) for g in s['gpus'])
 strikes=strikes+1 if bad else 0
 if s['available_gib']<16 or strikes>=3:raise RuntimeError('RESOURCE_PRESSURE_STOP '+json.dumps(s))
def launch(args,name,env):
 p=subprocess.Popen(args,cwd=W,env=env,stdout=(OUT/(name+'.log')).open('w'),stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run_eval(kind,checkpoint,run,dataset='pilot4',time_conditioned=False):
 check();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(W/'openpi_cache'))
 port=19035;python=str(W/'vendor/openpi/.venv/bin/python') if kind=='pi05' else PY
 server=launch([python,str(W/(kind+'_policy_server.py')),'--port',str(port)],run+'_server',env)
 deadline=time.monotonic()+300
 while True:
  if server.poll() is not None:raise RuntimeError('server exited')
  try:
   with socket.create_connection(('127.0.0.1',port),timeout=1):break
  except OSError:
   if time.monotonic()>deadline:raise TimeoutError('server startup')
   time.sleep(5);check()
 extra=['--time-conditioned','--evaluation-label','Four demonstrations, elapsed-time conditioned ACT overfit; no generalization claim'] if time_conditioned else []
 evaluation=launch([PY,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(checkpoint),'--run',run,'--dataset',dataset,'--gpu','1','--port',str(port)]+extra,run+'_eval',dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
 deadline=time.monotonic()+3*3600
 while evaluation.poll() is None:
  time.sleep(10);check()
  if server.poll() is not None:raise RuntimeError('server exited during evaluation')
  if time.monotonic()>deadline:raise TimeoutError('evaluation deadline')
 if evaluation.returncode:raise RuntimeError('evaluation failed; inspect log')
 stop(server)
try:
 check()
 s=snapshot()
 if s['available_gib']<64 or any(g[1]<20000 for g in s['gpus'] if g[0] in (0,1)):raise RuntimeError('Insufficient startup headroom')
 if True:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
  run='act_progress_v1'
  train=launch([PY,str(W/'train_progress_act.py'),'--dataset','pilot4_one_each','--run',run,'--steps','8000','--batch','4','--lr','3e-5','--cosine','--phase-balanced','--deterministic-latent','--time-conditioned','--resume',str(W/'runs/act_recovery_time/step_015000.pt')],run+'_train',env)
  deadline=time.monotonic()+4*3600
  while train.poll() is None:
   time.sleep(10);check()
   if time.monotonic()>deadline:raise TimeoutError('training deadline')
  if train.returncode:raise RuntimeError('training failed')
  run_eval('progress_act',W/'runs'/run/'step_008000.pt','act_progress_v1_8000','pilot4_one_each',True)
  marker='time_baseline_finished.json'
 else:
  if '--act-only' not in sys.argv:run_eval('pi05',W/'checkpoints/g2_pi05_lora/pilot4/3000','recovery_pi05_3000')
  run_eval('act',W/'runs/act_pilot4_robust/step_012000.pt','recovery_act_robust_12000')
  marker='finished.json'
 (OUT/marker).write_text(json.dumps(dict(time=time.strftime('%FT%T'),status='evaluations_complete')))
except Exception as e:
 (OUT/'stopped.json').write_text(json.dumps(dict(time=time.strftime('%FT%T'),error=repr(e))))
 raise
finally:
 for p in reversed(children):stop(p)
