"""Run two fresh repeats only after an initial physical trial passes."""
from pathlib import Path
import json,subprocess,sys,os,time
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent;aid,variant,pd,gpu2,gpu3=sys.argv[1:]
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 winners=[]
 for p in sorted((R/f'strategy15_{variant}_trials'/pd/aid).glob('candidate*_repeat1/result.json')):
  d=json.loads(p.read_text())
  if not d['success']:continue
  if d.get('plan',{}).get('mode','').startswith('bimanual_'):
   samples=json.loads(p.with_name('samples.json').read_text());both=sum(all(sum(f for n,f in r['finger_forces'].items() if n.startswith('gripper_'+s+'_'))>.5 for s in ['l','r']) for r in samples if r['phase']=='push')
   if both<5:continue
  winners.append(d)
 if winners:break
 time.sleep(5)
else:raise RuntimeError('No passing first trial appeared')
c=winners[0]['candidate'];print('REPEAT_BASELINE',aid,variant,c,flush=True)
def run(pair):
 repeat,gpu=pair
 with (R/f'{variant}_repeat_{aid}_{repeat}.log').open('w') as log:
  subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/f'{variant}_trial_strategy15.py'),'--asset',aid,'--plan-dir',pd,'--candidate',str(c),'--repeat',str(repeat),'--gpu',gpu],cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((R/f'strategy15_{variant}_trials'/pd/aid/f'candidate{c}_repeat{repeat}/result.json').read_text());print({k:d.get(k) for k in ['asset','repeat','success','failure','push_forward_m','max_yaw_error_deg','max_push_tilt_deg']},flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[(2,gpu2),(3,gpu3)]))
