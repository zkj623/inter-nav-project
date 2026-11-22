from pathlib import Path
import json,subprocess,concurrent.futures,threading,time,sys,os,random,hashlib
R=Path(__file__).resolve().parent;PY=sys.executable;cfg=json.loads((R/'standby_preparation_config.json').read_text());assets=list(cfg['execution']);lock=threading.Lock();reserved=set();audit_running=set();stopping=False;log=R/'collection_pipeline.jsonl';started=time.time();max_attempts=60;gpu_workers=[int(x) for x in os.environ.get('G2_COLLECTION_GPUS','2,6').split(',')]

def event(row):
 row={'wall_time':time.time(),**row}
 with lock:
  with log.open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)

def files():return list(R.glob('collection_trials/**/result.json'))
def status():
 counts={a:set() for a in assets};attempts={a:0 for a in assets};pending={a:0 for a in assets};failures={};unverified=[]
 for f in files():
  try:d=json.loads(f.read_text())
  except Exception:continue
  a=d['asset'];attempts[a]+=1;v=f.with_name('verification.json')
  if v.exists():
   try:r=json.loads(v.read_text())
   except Exception:continue
   if r['verified_success']:counts[a].add(d['plan']['collection_placement']['group_id'])
   else:failures[r.get('failure') or '|'.join(r.get('rejections',[]))]=failures.get(r.get('failure') or '|'.join(r.get('rejections',[])),0)+1
  else:
   unverified.append(f)
   if d['success']:pending[a]+=1
 return counts,attempts,pending,failures,unverified

def audit(f):
 D=f.parent
 try:
  p=subprocess.run([PY,str(R/'verify_collection.py'),'--trial-dir',str(D)],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (D/'independent_audit.log').write_text(p.stdout)
  event(dict(type='audit',trial=str(D),code=p.returncode,tail=p.stdout[-1400:]))
 finally:
  with lock:audit_running.discard(str(f))

def audit_monitor():
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  while not stopping:
   _,_,_,_,todo=status()
   for f in todo:
    with lock:
     if str(f) in audit_running:continue
     audit_running.add(str(f))
    pool.submit(audit,f)
   time.sleep(4)

def existing_gpu(gpu):
 ps=subprocess.check_output(['ps','-u','zkj','-o','args='],text=True)
 return any(('python ' in line or '/python ' in line) and str(R) not in line.split('python ')[0] and ('collect_' in line or 'replay_' in line) and '_trial.py' in line and ('--gpu '+str(gpu)+' ' in line+' ') and 'bash -c' not in line for line in ps.splitlines())

def completed_indices(a):
 return {int(f.parts[-4].split('_')[-1]) for f in R.glob(f'collection_trials/collection_plans/placement_*/{a}/candidate0_repeat*/result.json')}

def choose():
 counts,attempts,pending,_,_=status();active_counts={a:sum(x[0]==a for x in reserved) for a in assets}
 order=sorted(assets,key=lambda a:(len(counts[a])+pending[a]+active_counts[a],attempts[a],assets.index(a)))
 for a in order:
  if len(counts[a])+pending[a]+active_counts[a]>=20 or attempts[a]>=max_attempts:continue
  for f in sorted(R.glob(f'collection_plans/placement_*/{a}_planning.json')):
   d=json.loads(f.read_text());i=d['index'];key=(a,i)
   if not d['success'] or key in reserved:continue
   trial=R/'collection_trials'/'collection_plans'/f'placement_{i:04d}'/a/'candidate0_repeat1'
   if trial.exists():continue
   reserved.add(key);return 'run',a,i
 # Generate narrower placements when wide randomization has exhausted reachable candidates.
 for a in order:
  if len(counts[a])+pending[a]+active_counts[a]>=20 or attempts[a]>=max_attempts:continue
  ids=[int(f.parent.name.split('_')[-1]) for f in R.glob(f'collection_plans/placement_*/{a}_planning.json')]
  i=max([24]+ids+[x[1] for x in reserved if x[0]==a])+1
  if i>160:continue
  reserved.add((a,i));return 'plan',a,i
 return None

def worker(gpu):
 while True:
  if existing_gpu(gpu):time.sleep(5);continue
  with lock:
   job=choose()
  if job is None:
   counts,attempts,pending,_,_=status()
   if all(len(counts[a])>=20 or attempts[a]>=max_attempts for a in assets):return
   if not reserved and not any(pending.values()):return
   time.sleep(5);continue
  kind,a,i=job
  if kind=='plan':
   rng=random.Random(20261122+i*71+assets.index(a))
   if a=='stool_rectangular':dx=rng.uniform(-.004,0.);dy=rng.uniform(-.003,.003);yaw=0.
   elif a=='table_side':dx=rng.uniform(-.004,.004);dy=rng.uniform(-.004,.004);yaw=rng.uniform(-.5,.5)
   elif a=='box_reinforced':dx=rng.uniform(.001,.006);dy=rng.uniform(-.003,.003);yaw=rng.uniform(-.5,.5)
   else:dx=rng.uniform(-.005,.008);dy=rng.uniform(-.005,.005);yaw=rng.uniform(-1.,1.)
   p=subprocess.run([PY,str(R/'plan_collection.py'),'--asset',a,'--index',str(i),'--dx',str(dx),'--dy',str(dy),'--yaw',str(yaw)],cwd=R,capture_output=True,text=True)
   event(dict(type='supplemental_plan',asset=a,index=i,code=p.returncode,output=p.stdout[-1500:]))
   with lock:reserved.discard((a,i))
   continue
  event(dict(type='launch',asset=a,index=i,gpu=gpu))
  outfile=R/'batch_logs'/f'{a}_{i:04d}.log';outfile.parent.mkdir(exist_ok=True)
  try:
   with outfile.open('w') as out:
    p=subprocess.run([PY,str(R/f'collect_{a}_trial.py'),'--asset',a,'--plan-dir',f'collection_plans/placement_{i:04d}','--gpu',str(gpu)],cwd=R,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
   event(dict(type='finished',asset=a,index=i,gpu=gpu,returncode=p.returncode))
  except Exception as exc:event(dict(type='process_error',asset=a,index=i,gpu=gpu,error=str(exc)))
  finally:
   with lock:reserved.discard((a,i))

def publish():
 counts,attempts,pending,failures,_=status();ds=R/'dataset';ds.mkdir(exist_ok=True);rows=[]
 for a in assets:
  valid=[]
  for f in R.glob(f'collection_trials/**/{a}/candidate*/verification.json'):
   r=json.loads(f.read_text())
   if r['verified_success']:valid.append((f,r))
  valid.sort(key=lambda x:str(x[0]));unique={}
  for f,r in valid:
   meta=json.loads(f.with_name('metadata.json').read_text());unique.setdefault(meta['placement']['group_id'],(f,r,meta))
  chosen=list(unique.values())[:20];permutation=list(range(len(chosen)));random.Random(20261122+assets.index(a)).shuffle(permutation)
  split={i:('train' if n<16 else 'validation' if n<18 else 'test') for n,i in enumerate(permutation)}
  for idx,(f,r,meta) in enumerate(chosen):
   target=ds/'accepted'/a/(meta['placement']['group_id']+'.hdf5');target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():os.link(f.with_name('episode.hdf5'),target)
   rows.append(dict(asset=a,group_id=meta['placement']['group_id'],episode=str(target.relative_to(ds)),split=split[idx],frames=r['data']['frames'],sha256=r['episode_sha256'],raw_trial=str(f.parent),dx_m=meta['placement']['dx_m'],dy_m=meta['placement']['dy_m'],dyaw_deg=meta['placement']['dyaw_deg'],source_sha256=meta['source_sha256'],verification=r))
 with (ds/'manifest.jsonl').open('w') as out:
  for row in rows:out.write(json.dumps(row)+'\n')
 summary=dict(updated_wall_time=time.time(),elapsed_seconds=time.time()-started,target_per_asset=20,total_verified=sum(len(x) for x in counts.values()),counts={a:len(counts[a]) for a in assets},attempts=attempts,pending=pending,failures=failures,running=not stopping,physics_hz=120,control_hz=20,root_pinned=True,torso='existing ideal1e7; locked at original common home',mass_kg=1.,dataset_manifest=str(ds/'manifest.jsonl'))
 (R/'collection_progress.json').write_text(json.dumps(summary,indent=2));return summary

monitor=threading.Thread(target=audit_monitor,daemon=True);monitor.start()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_workers)) as pool:
  tasks=[pool.submit(worker,gpu) for gpu in gpu_workers]
  while not all(t.done() for t in tasks):
   print('PROGRESS '+json.dumps(publish()),flush=True);time.sleep(20)
  for t in tasks:t.result()
 while audit_running or status()[-1]:publish();time.sleep(5)
finally:
 stopping=True;monitor.join(timeout=10);print('FINAL '+json.dumps(publish()),flush=True)

