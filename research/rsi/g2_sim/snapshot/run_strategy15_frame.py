from pathlib import Path
import os,sys,json,subprocess,time,threading,queue,concurrent.futures,traceback,shutil
R=Path(__file__).resolve().parent;L=R/'strategy15_logs';env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':str(R.parent/'r1lite-collision-deps')}
tasks={'chair_dining':'body','chair_wheeled':'body','stool_wood_four_leg':'body'}
records={a:dict(asset=a,status='queued',trials=[]) for a in tasks};lock=threading.Lock();jobs=queue.Queue()
def save(a,**v):
 with lock:
  records[a].update(v);(R/'strategy15_frame_status.json').write_text(json.dumps(list(records.values()),indent=2));print(json.dumps({'asset':a,**v}),flush=True)
def run(script,aid,extra=(),timeout=1500,tag=None):
 with (L/f'{aid}_retry_{tag or script}.log').open('w') as log:
  args=['--asset',aid] if script.startswith(('plan_','grasp_')) else [aid]
  p=subprocess.run([sys.executable,str(R/script),*args,*extra],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError(f'{script} exited {p.returncode}')
def prepare(aid):
 try:
  mode=tasks[aid];save(aid,status='planning',mode=mode)
  if mode=='body':run('plan_strategy15_frame.py',aid,timeout=700)
  elif mode=='feature':run('plan_strategy15_feature_side.py',aid,timeout=700)
  elif mode=='existing_body':
   for _ in range(80):
    if (R/'strategy15_frame_plans'/(aid+'.json')).exists():break
    time.sleep(10)
   else:raise RuntimeError('Existing planner did not complete')
  elif mode=='existing_feature':
   for _ in range(80):
    processes=subprocess.check_output(['pgrep','-af','plan_strategy15_feature_side.py'],text=True)
    if not any('--asset '+aid in l for l in processes.splitlines()):break
    time.sleep(10)
  if 'feature' in mode:
   target=R/'strategy15_frame_plans'/(aid+'.json');target.parent.mkdir(exist_ok=True);shutil.copy2(R/'strategy15_feature_side_plans'/(aid+'.json'),target)
  save(aid,status='auditing')
  for script in ('strategy15_frame_validate_plans.py','strategy15_frame_descent.py','strategy15_frame_audit_native_self.py','strategy15_frame_audit_object_clearance.py'):run(script,aid)
  path=R/'strategy15_frame_validated_plans'/(aid+'.json');d=json.loads(path.read_text());d['plans']=[p for p in d['plans'] if 'q_start' in p];path.write_text(json.dumps(d,indent=2))
  save(aid,status='ready' if d['plans'] else 'no_admissible_plan',validated_candidates=len(d['plans']))
  if d['plans']:jobs.put(aid)
 except Exception:save(aid,status='pipeline_error',error=traceback.format_exc())
def free(gpu):
 s=json.loads((R/'strategy15_status.json').read_text())
 if any(r['status'] in ('queued','planning','auditing','ready') for r in s):return False
 ps=subprocess.check_output(['ps','-eo','args'],text=True)
 return not any('grasp_trial_strategy15' in l and f'--gpu {gpu}' in l and not l.startswith('bash ') for l in ps.splitlines())
def physical(gpu):
 while True:
  aid=jobs.get()
  if aid is None:jobs.task_done();return
  trials=[]
  try:
   while not free(gpu):time.sleep(10)
   save(aid,status='simulating',gpu=gpu);plans=json.loads((R/'strategy15_frame_validated_plans'/(aid+'.json')).read_text())['plans']
   for c in range(min(3,len(plans))):
    passed=False
    for rep in (1,2,3):
     run('grasp_trial_strategy15_frame.py',aid,('--candidate',str(c),'--repeat',str(rep),'--gpu',str(gpu)),tag=f'c{c}r{rep}')
     path=R/f'strategy15_frame_trials/strategy15_frame_validated_plans/{aid}/candidate{c}_repeat{rep}/result.json';d=json.loads(path.read_text());record={k:d.get(k) for k in ('candidate','repeat','success','failure','error','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','max_contact_surface_error_m')};record.update(path=str(path.relative_to(R)),strategy=plans[c].get('strategy'));trials.append(record);save(aid,trials=trials.copy())
     if d['failure']=='HARNESS_ERROR':raise RuntimeError(d.get('error'))
     if not d['success']:break
     passed=True
    if passed:break
   save(aid,status='tested')
  except Exception:save(aid,status='pipeline_error',error=traceback.format_exc())
  jobs.task_done()
threads=[threading.Thread(target=physical,args=(g,)) for g in (6,)]
for t in threads:t.start()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(prepare,tasks))
for _ in threads:jobs.put(None)
for t in threads:t.join()
print('STRATEGY15_TILT_COMPLETE',flush=True)
