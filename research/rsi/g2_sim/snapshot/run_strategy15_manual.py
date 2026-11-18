from pathlib import Path
import subprocess,sys,json,os,shutil,time
R=Path(__file__).resolve().parent;env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':str(R.parent/'r1lite-collision-deps')}
aid=sys.argv[1];gpu=sys.argv[2];variant=sys.argv[3]
if aid=='stool_wood_four_leg' and variant=='retry':
 repaired=R/'strategy15_surface_trials/strategy15_surface_validated_plans/stool_wood_four_leg/candidate0_repeat1/result.json'
 if repaired.exists() and json.loads(repaired.read_text()).get('success'):
  print('SUPERSEDED_BY_VALIDATED_SURFACE_REPAIR',flush=True);sys.exit(0)
if aid=='stool_wood_four_leg' and variant=='retry':
 shutil.copy2(R/'strategy15_feature_side_plans'/f'{aid}.json',R/'strategy15_retry_plans'/f'{aid}.json')
 for name in ['validate_plans','descent','audit_native_self','audit_object_clearance']:
  with (R/'strategy15_logs'/f'{aid}_manual_{name}.log').open('w') as f:subprocess.run([sys.executable,str(R/f'strategy15_retry_{name}.py'),aid],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
pd=f'strategy15_{variant}_validated_plans';plans=json.loads((R/pd/f'{aid}.json').read_text())['plans']
for c in range(min(4,len(plans))):
 passed=[]
 for rep in (1,2,3):
  with (R/'strategy15_logs'/f'{aid}_{variant}_c{c}r{rep}_manual.log').open('w') as f:subprocess.run([sys.executable,str(R/f'grasp_trial_strategy15_{variant}.py'),'--asset',aid,'--candidate',str(c),'--repeat',str(rep),'--gpu',gpu],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  p=R/f'strategy15_{variant}_trials'/pd/aid/f'candidate{c}_repeat{rep}'/'result.json';d=json.loads(p.read_text());print(aid,c,rep,d['success'],d.get('failure'),flush=True);passed.append(d['success'])
  if not d['success']:break
 if len(passed)==3 and all(passed):break
