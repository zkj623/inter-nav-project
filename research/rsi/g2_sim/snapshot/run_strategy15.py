from pathlib import Path
import os,sys,json,subprocess,time,threading,queue,concurrent.futures,traceback
R=Path(__file__).resolve().parent;L=R/'strategy15_logs';L.mkdir(exist_ok=True)
ids=json.loads((R/'rigid_catalog.json').read_text())['active_assets']
ids.sort(key=lambda a:(a not in ('bin_pedal','bin_round_lidded','box_white','basket_woven'),a))
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':str(R.parent/'r1lite-collision-deps')}
lock=threading.Lock();jobs=queue.Queue();records={a:dict(asset=a,status='queued',trials=[]) for a in ids}
def save(a,**v):
 with lock:
  records[a].update(v);(R/'strategy15_status.json').write_text(json.dumps(list(records.values()),indent=2));print(json.dumps({'asset':a,**v}),flush=True)
def run(script,aid,extra=(),timeout=1200,tag=None):
 with (L/f'{aid}_{tag or script}.log').open('w') as log:
  args=['--asset',aid] if script.startswith(('plan_','grasp_')) else [aid]
  p=subprocess.run([sys.executable,str(R/script),*args,*extra],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError(f'{script}: returncode {p.returncode}')
def prepare(aid):
 try:
  save(aid,status='planning');docs={}
  order=('feature','spread') if aid in ('basket_woven','basket_rectangular','stool_rectangular','stool_wood_four_leg','table_side','chair_dining','chair_wheeled') else ('spread','feature')
  for mode in order:
   run(f'plan_strategy15_{mode}.py',aid,timeout=650)
   docs[mode]=json.loads((R/f'strategy15_{mode}_plans/{aid}.json').read_text())
   save(aid,status='planning',last_planner=mode,candidates_so_far=sum(len(d['plans']) for d in docs.values()))
  plans=[]
  for i in range(6):
   for mode in order:
    if i<len(docs[mode]['plans']):plans.append(docs[mode]['plans'][i])
  # Previously validated configurations are explicit regression candidates, never
  # silently presented as new spread-contact successes.
  old=R/('front_dual_validated_plans' if aid=='basket_woven' else 'catalog_dual_validated_plans')/(aid+'.json')
  if aid in ('basket_woven','bin_pedal','bin_round_lidded') and old.exists():
   baseline=json.loads(old.read_text())['plans'][:1]
   for p in baseline:p['strategy']='previous_validated_regression';p['hold_gripper_q']=0.
   plans+=baseline
  (R/'strategy15_plans').mkdir(exist_ok=True);(R/'strategy15_plans'/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=plans,searches={m:{k:v for k,v in d.items() if k!='plans'} for m,d in docs.items()}),indent=2))
  save(aid,status='auditing',raw_candidates=len(plans))
  for script in ('strategy15_validate_plans.py','strategy15_descent.py','strategy15_audit_native_self.py','strategy15_audit_object_clearance.py'):run(script,aid)
  path=R/'strategy15_validated_plans'/(aid+'.json');d=json.loads(path.read_text());d['plans']=[p for p in d['plans'] if 'q_start' in p];path.write_text(json.dumps(d,indent=2))
  save(aid,status='ready' if d['plans'] else 'no_admissible_plan',validated_candidates=len(d['plans']))
  if d['plans']:jobs.put(aid)
 except Exception:save(aid,status='pipeline_error',error=traceback.format_exc())
def physical(gpu):
 while True:
  aid=jobs.get()
  if aid is None:jobs.task_done();return
  trials=[]
  try:
   save(aid,status='simulating',gpu=gpu);plans=json.loads((R/'strategy15_validated_plans'/(aid+'.json')).read_text())['plans']
   candidates=[]
   for strategy in dict.fromkeys(p.get('strategy','unknown') for p in plans):
    candidates.append(next(i for i,p in enumerate(plans) if p.get('strategy','unknown')==strategy))
   for i in range(len(plans)):
    if len(candidates)>=4:break
    if i not in candidates:candidates.append(i)
   def trial(c,rep):
    run('grasp_trial_strategy15.py',aid,('--candidate',str(c),'--repeat',str(rep),'--gpu',str(gpu)),timeout=1200,tag=f'c{c}r{rep}')
    path=R/f'strategy15_trials/strategy15_validated_plans/{aid}/candidate{c}_repeat{rep}/result.json';d=json.loads(path.read_text());record={k:d.get(k) for k in ('candidate','repeat','success','failure','error','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','max_contact_surface_error_m')};record.update(path=str(path.relative_to(R)),strategy=plans[c].get('strategy'));trials.append(record);save(aid,trials=trials.copy());return d
   for c in candidates:
    d=trial(c,1)
    if d['failure']=='HARNESS_ERROR':raise RuntimeError(d.get('error'))
    if d['success']:
     for rep in (2,3):trial(c,rep)
     break
   save(aid,status='tested')
  except Exception:save(aid,status='pipeline_error',error=traceback.format_exc())
  jobs.task_done()
threads=[threading.Thread(target=physical,args=(g,)) for g in (2,4,6)]
for t in threads:t.start()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(prepare,ids))
for t in threads:jobs.put(None)
for t in threads:t.join()
print('STRATEGY15_COMPLETE',flush=True)
