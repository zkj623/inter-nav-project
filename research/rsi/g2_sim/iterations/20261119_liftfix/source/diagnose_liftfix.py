from pathlib import Path
import json
R=Path(__file__).resolve().parent
for p in sorted(R.glob('liftfix*_trials/**/result.json')):
 d=json.loads(p.read_text())
 if d.get('success') or d.get('failure')=='HARNESS_ERROR':continue
 print(str(p.relative_to(R)),d.get('failure'),flush=True)
 nc=p.with_name('nonhand_contacts.json')
 if nc.exists():
  hits=[x for x in json.loads(nc.read_text()) if x['impulse']>.001]
  print('nonhand',sorted(hits,key=lambda x:-x['impulse'])[:3],flush=True)
 ss=json.loads(p.with_name('samples.json').read_text())
 for s in ss[-2:]:
  forces={}
  for side in ('l','r'):
   for finger in ('inner','outer'):
    forces[side+'_'+finger]=sum(v for n,v in s['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_'))
  print(dict(t=s['t'],phase=s['phase'],lift=s['lift_m'],tilt=s['tilt_deg'],forces=forces),flush=True)
