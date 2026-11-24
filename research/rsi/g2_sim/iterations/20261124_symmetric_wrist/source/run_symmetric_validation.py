from pathlib import Path
import json,subprocess,sys,concurrent.futures,queue,copy,time
R=Path(__file__).resolve().parent;cfg=json.loads((R/'ground_lift_training_config.json').read_text());prep=json.loads((R/'standby_preparation_config.json').read_text());gpuq=queue.Queue()
for g in [2,6]:gpuq.put(g)
P=R/'symmetric_preparation_plans';P.mkdir(exist_ok=True);doc=json.loads((R/'realpose_plans/case_handheld.json').read_text());p=doc['plans'][0];p['q_default']=prep['standby_joint_positions_rad'];p['q_prepare']=[p['q_start']];p['collection_initialization']='full standby to symmetric wrist low home then approach/grasp/lift';(P/'case_handheld.json').write_text(json.dumps(doc,indent=2))
def run(a):
 gpu=gpuq.get();folder='symmetric_preparation_plans' if a=='case_handheld' else 'realpose_plans'
 try:
  with (R/f'symmetric_{a}.log').open('w') as log:result=subprocess.run([sys.executable,str(R/f'realpose_{a}_trial.py'),'--asset',a,'--plan-dir',folder,'--gpu',str(gpu)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  trial=R/'realpose_trials'/folder/a/'candidate0_repeat1'
  if not (trial/'result.json').exists():return dict(asset=a,verified_success=False,error='missing result',exit_code=result.returncode)
  with (R/f'symmetric_{a}_audit.log').open('w') as log:subprocess.run([sys.executable,str(R/'verify_realpose.py'),'--trial-dir',str(trial)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  v=trial/'verification.json';r=json.loads(v.read_text()) if v.exists() else dict(asset=a,verified_success=False,error='audit failed');print('VERIFIED',a,r.get('verified_success'),r.get('failure'),r.get('rejections'),flush=True);return r
 finally:gpuq.put(gpu)
assets=list(prep['execution']);assets.sort(key=lambda a:0 if a=='case_handheld' else 1);rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for future in concurrent.futures.as_completed([pool.submit(run,a) for a in assets]):
  try:r=future.result()
  except Exception as e:r=dict(verified_success=False,error=str(e))
  rows.append(r);(R/'symmetric_validation.json').write_text(json.dumps(rows,indent=2))
print('ALL_DONE',sum(r.get('verified_success',False) for r in rows),len(rows),flush=True)
