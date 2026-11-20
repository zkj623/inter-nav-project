from pathlib import Path
import json,subprocess,os,concurrent.futures
R=Path(__file__).resolve().parent;PY='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python';cfg=json.loads((R/'ground_lift_training_config.json').read_text());assets=list(cfg['validated_simulation_baselines']);priority=['box_white','case_handheld','basket_rectangular','stool_wood_four_leg'];assets=priority+[a for a in assets if a not in priority]
def worker(gpu,items):
 for a in items:
  p=R/'symhome_plans'/(a+'.json')
  if not p.exists() or not json.loads(p.read_text())['plans']:print('SKIP',a,flush=True);continue
  with (R/f'symhome_v3_{a}.log').open('w') as f:
   r=subprocess.run([PY,str(R/f'symhome_v3_{a}_trial.py'),'--asset',a,'--plan-dir','symhome_plans','--gpu',str(gpu)],cwd=R,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
  print('FINISHED',a,r.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(lambda x:worker(*x),[(g,assets[i::4]) for i,g in enumerate([6,5,2,4])]))
print('SYMHOME_BATCH_DONE',flush=True)
