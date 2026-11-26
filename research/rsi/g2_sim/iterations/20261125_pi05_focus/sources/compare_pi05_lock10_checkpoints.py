from pathlib import Path
import os,sys,json,time,subprocess,signal,socket,shutil,fcntl,math
W=Path('/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124')
OUT=W/'pi05_focus_20261125';OUT.mkdir(exist_ok=True)
PY='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
JP=str(W/'vendor/openpi/.venv/bin/python')
FAST=Path('/tmp/zkj_g2_20261125_checkpoints')
GPU=2;SIM_GPU=5;PORT=19045
children=[];previous=None;strikes=0;gpu_cooldowns={5:time.monotonic()+1800}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
def status(phase,**kw):
 d=dict(time=time.strftime('%FT%T'),phase=phase,**kw)
 (OUT/'status.tmp.json').write_text(json.dumps(d,indent=2));(OUT/'status.tmp.json').replace(OUT/'status.json')
 print(json.dumps(d),flush=True)
def check(start=False,gpus=None):
 global previous,strikes
 if gpus is None:gpus=(GPU,)
 m={k:int(v.split()[0]) for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
 v=dict(l.split() for l in Path('/proc/vmstat').read_text().splitlines());c=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));t=time.monotonic();sw=int(v['pswpin'])+int(v['pswpout']);rate=wait=0.
 if previous:
  dt=max(t-previous[0],.01);rate=(sw-previous[1])*os.sysconf('SC_PAGE_SIZE')/dt/2**20;dc=sum(c)-sum(previous[2]);wait=100*(c[4]-previous[2][4])/max(dc,1)
 previous=(t,sw,c)
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 gpu={int(x.split(',')[0]):[int(y) for y in x.split(',')[1:]] for x in raw.strip().splitlines()}
 s=dict(time=time.strftime('%FT%T'),available_gib=m['MemAvailable']/2**20,swap_io_mib_s=rate,swap_used_mib=(m['SwapTotal']-m['SwapFree'])/1024,iowait_pct=wait,root_free_gib=shutil.disk_usage('/').free/2**30,data_free_gib=shutil.disk_usage(W).free/2**30,gpus=gpu)
 with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
 bad=s['available_gib']<32 or rate>20 or wait>30 or s['root_free_gib']<45 or s['data_free_gib']<100 or any(gpu[i][0]<1536 or gpu[i][1]>85 for i in gpus)
 strikes=strikes+1 if bad else 0
 if s['available_gib']<16 or strikes>=3:raise RuntimeError('RESOURCE_PRESSURE_STOP '+json.dumps(s))
 if start and (s['available_gib']<64 or s['root_free_gib']<80 or any(gpu[i][0]<18000 for i in gpus)):raise RuntimeError('INSUFFICIENT_STARTUP_HEADROOM '+json.dumps(s))
 return s
def launch(args,label,env=None):
 e=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_VISIBLE_DEVICES=str(GPU),WANDB_MODE='disabled',OPENPI_DATA_HOME=str(W/'openpi_cache'))
 if env:e.update(env)
 p=subprocess.Popen(args,cwd=W,env=e,stdout=(OUT/(label+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True);children.append(p)
 return p
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def wait(p,timeout,gpus=None):
 deadline=time.monotonic()+timeout
 while p.poll() is None:
  time.sleep(10);check(gpus=gpus)
  if time.monotonic()>deadline:raise TimeoutError('Job deadline exceeded')
 if p.returncode:raise RuntimeError('Child failed '+str(p.args)+' rc='+str(p.returncode))
def await_capacity(operation,requirements):
 deadline=time.monotonic()+12*3600
 while True:
  current=check(gpus=())
  ready=current['available_gib']>=64 and current['root_free_gib']>=80 and all(current['gpus'][gpu][0]>=need and current['gpus'][gpu][1]<=82 for gpu,need in requirements.items())
  if ready:return
  status('waiting_resources',operation=operation,required_free_mib=requirements,available_gib=current['available_gib'],gpu_free_mib={gpu:current['gpus'][gpu][0] for gpu in requirements})
  if time.monotonic()>deadline:raise TimeoutError('Resource wait exceeded 12 hours')
  time.sleep(10)
def select_roles(operation):
 global GPU,SIM_GPU
 deadline=time.monotonic()+12*3600
 while True:
  current=check(gpus=());ranked=sorted(current['gpus'],key=lambda i:current['gpus'][i][0],reverse=True)
  system_ok=current['available_gib']>=64 and current['root_free_gib']>=80
  candidates=[i for i in ranked if current['gpus'][i][1]<=82 and time.monotonic()>=gpu_cooldowns.get(i,0)]
  if system_ok and operation=='training' and any(current['gpus'][i][0]>=18000 for i in candidates):
   GPU=next(i for i in candidates if current['gpus'][i][0]>=18000)
   status('resources_ready',operation=operation,training_gpu=GPU,free_mib=current['gpus'][GPU][0]);return
  if system_ok and operation=='evaluation':
   pairs=[(i,j) for i in candidates for j in candidates if i!=j and current['gpus'][i][0]>=12000 and current['gpus'][j][0]>=10000]
   if pairs:
    GPU,SIM_GPU=pairs[0];status('resources_ready',operation=operation,policy_gpu=GPU,simulation_gpu=SIM_GPU);return
  status('waiting_resources',operation=operation,available_gib=current['available_gib'],gpu_free_mib={i:current['gpus'][i][0] for i in ranked},required_free_mib=18000 if operation=='training' else {'policy':12000,'simulation':10000})
  if time.monotonic()>deadline:raise TimeoutError('Resource wait exceeded 12 hours')
  time.sleep(10)
def train(run,dataset,steps,initial):
 global strikes
 logical=W/'checkpoints/g2_pi05_lora'/run;physical=FAST/'g2_pi05_lora'/run;physical.parent.mkdir(parents=True,exist_ok=True)
 if not logical.exists() and not logical.is_symlink():logical.symlink_to(physical,target_is_directory=True)
 assert logical.is_symlink() and logical.resolve()==physical
 for attempt in range(1,5):
  select_roles('training')
  status('training',run=run,dataset=dataset,target_steps=steps,batch_size=2,init_checkpoint=str(initial),attempt=attempt,gpu=GPU)
  p=launch([JP,str(W/'train_pi05_focus_long_relative.py'),'--run',run,'--dataset',dataset,'--steps',str(steps),'--batch','2','--init-checkpoint',str(initial/'params')],run+'_train_'+str(steps))
  status('training',run=run,dataset=dataset,target_steps=steps,batch_size=2,pid=p.pid,attempt=attempt,gpu=GPU)
  try:
   wait(p,max(4*3600,steps*1.5))
   break
  except RuntimeError as ex:
   if 'RESOURCE_PRESSURE_STOP' not in str(ex) or attempt==4:raise
   stop(p);gpu_cooldowns[GPU]=time.monotonic()+1800;strikes=0
   status('retrying_after_resource_pressure',run=run,target_steps=steps,attempt=attempt,avoided_gpu=GPU,reason=str(ex),recovery='Resume latest completed checkpoint; no changes to unrelated jobs')
 paths=[p for p in physical.iterdir() if p.name.isdigit() and (p/'params').exists() and (p/'_CHECKPOINT_METADATA').exists()]
 if not paths:raise RuntimeError('No completed checkpoint after training')
 cp=max(paths,key=lambda p:int(p.name));assert int(cp.name)>=steps-1
 status('checkpoint_complete',run=run,checkpoint=str(cp))
 return cp
def evaluate(cp,run,dataset,scope,split='train',seed=20261125):
 select_roles('evaluation')
 server=launch([JP,str(W/'pi05_policy_server.py'),'--port',str(PORT),'--scope',scope],run+'_server')
 try:
  deadline=time.monotonic()+300
  while True:
   if server.poll() is not None:raise RuntimeError('Policy server exited')
   try:
    with socket.create_connection(('127.0.0.1',PORT),timeout=1):break
   except OSError:
    if time.monotonic()>deadline:raise TimeoutError('Server startup')
    time.sleep(5);check()
  status('evaluating',run=run,checkpoint=str(cp),dataset=dataset,split=split)
  env={'CUDA_VISIBLE_DEVICES':','.join(map(str,range(8)))}
  p=launch([PY,str(W/'evaluate_pi05_paired_lock10.py'),'--checkpoint',str(cp),'--run',run,'--dataset',dataset,'--gpu',str(SIM_GPU),'--port',str(PORT),'--protocol','lock10','--execute','5','--sampling-seed',str(seed),'--split',split,'--evaluation-label','pi05 language+3RGB+qpos; lock10_lift_v1; '+split],run+'_eval',env)
  wait(p,4*3600,gpus=(GPU,SIM_GPU))
  result=json.loads((W/'evaluations'/run/'summary.json').read_text());status('evaluation_complete',run=run,successes=result['successes'],total=result['total'])
  return result
 finally:stop(server)
def backup(cp,label):
 status('backing_up',checkpoint=str(cp))
 dest=W/'checkpoint_backups/pi05_focus'/label
 p=launch([PY,str(W/'backup_focus_checkpoint.py'),str(cp),str(dest)],'backup_'+label)
 wait(p,3600,gpus=())

import html
comparison_lock=(OUT/'paired_lock10.lock').open('w');fcntl.flock(comparison_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=json.loads((OUT/'status.json').read_text())
assert prior['phase']=='training',prior
TRAINER=prior['pid'];SUPERVISOR=json.loads((OUT/'supervisor_pid.json').read_text())['pid']
for pid,needle in [(SUPERVISOR,'pi05_focus_long_supervisor.py'),(TRAINER,'train_pi05_focus_long_relative.py')]:
 assert needle in Path(f'/proc/{pid}/cmdline').read_text()
assert os.getpgid(TRAINER)==TRAINER
PORT=19047;paused=False;supervisor_paused=False;results=[]
manifest=json.loads((OUT/'paired_lock10_checkpoints.json').read_text())
def interrupted(sig,frame):raise RuntimeError('COMPARISON_INTERRUPTED_'+str(sig))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def write_report(done=False):
 report=dict(protocol='lock10_lift_v1',seed_reset_per_asset=True,sampling_seed=20261125,complete=done,expected_checkpoints=[86000,129000,140000],expected_assets=['box_white','bin_lidded','bin_pedal','basket_rectangular'],stages=results,notes='One rollout per training object and checkpoint, paired initial conditions and per-object random seed. Runtime failures are separate from policy failures. No transport or robustness claim.')
 scores=[(r['checkpoint_label'],sum(x['success'] for x in r['results'])) for r in results]
 report['score_by_checkpoint']=dict(scores)
 if done and scores and all(r['total']==4 and not any(x.get('failure')=='EVALUATION_NO_RESULT' for x in r['results']) for r in results):
  best=max(v for _,v in scores);report['best_checkpoint_labels']=[k for k,v in scores if v==best]
 (OUT/'paired_lock10_comparison.json').write_text(json.dumps(report,indent=2))
 parts=['<!doctype html><meta charset="utf-8"><title>Paired checkpoint evaluation</title><style>body{font:17px sans-serif;max-width:1100px;margin:30px auto}td,th{padding:10px;border-bottom:1px solid #ddd}video{max-width:800px;width:100%}</style><h1>Paired lock10 evaluation</h1><p>Same placements, initialization and per-object seed. Stop policy at 10 cm; hold targets 5 seconds. Base contact is diagnostic only.</p>']
 for stage in results:
  parts.append('<h2>'+str(stage['checkpoint_label'])+' steps</h2><table><tr><th>Object</th><th>Outcome</th><th>Maximum lift cm</th></tr>')
  for x in stage['results']:
   parts.append('<tr><td>'+x['asset']+'</td><td>'+html.escape('PASS' if x['success'] else str(x.get('failure')) )+'</td><td>'+str(round(100*(x.get('max_lift_m') or 0),2))+'</td></tr>')
  parts.append('</table><a href="reviews/'+stage['run']+'/index.html">Videos and details</a>')
 parts.append('<p>Complete: '+str(done)+'</p>');(OUT/'paired_lock10_comparison.html').write_text(''.join(parts))
try:
 check(gpus=())
 os.kill(SUPERVISOR,signal.SIGSTOP);supervisor_paused=True
 os.killpg(TRAINER,signal.SIGSTOP);paused=True
 status('paired_checkpoint_comparison',trainer_paused=TRAINER,supervisor_paused=SUPERVISOR,checkpoints=[86000,129000,140000])
 write_report()
 for entry in manifest:
  cp=Path(entry['checkpoint']);run='pi05_focus_paired_relative_'+str(entry['label'])+'_lock10'
  result=evaluate(cp,run,'pilot4','pi05_pilot8_relative_focus_v1_scope.json')
  result['checkpoint_label']=entry['label'];results.append(result);write_report()
 write_report(True)
 print('PAIRED_COMPARISON_COMPLETE',flush=True)
except Exception as ex:
 (OUT/'paired_lock10_error.json').write_text(json.dumps(dict(error=repr(ex),time=time.strftime('%FT%T'))))
 raise
finally:
 for child in reversed(children):stop(child)
 if paused:os.killpg(TRAINER,signal.SIGCONT)
 if supervisor_paused:os.kill(SUPERVISOR,signal.SIGCONT)
 status('training',**{k:v for k,v in prior.items() if k not in ('time','phase','manual_evaluation_run','manual_evaluation_finished')},manual_evaluation_run='paired_lock10_comparison',manual_evaluation_finished=time.strftime('%FT%T'))
