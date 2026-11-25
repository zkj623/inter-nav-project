from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';D=R/'source_versions';D.mkdir(exist_ok=True);registered={}
for asset in ['box_white','bin_lidded','bin_pedal','basket_rectangular']:
 p=R/f'policy_force_{asset}_trial.py';data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();trial=R/'policy_trials/act_force_v1_8000/placements/p0001'/asset/'candidate0_repeat1/result.json'
 if trial.exists():assert json.loads(trial.read_text())['source_sha256']==sha,'Do not register changed source for an existing result'
 target=D/(sha+'.py')
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 registered[asset]=dict(source=str(p),sha256=sha,archive=str(target))
(W/'force_source_registration.json').write_text(json.dumps(registered,indent=2));print('EXACT_SOURCE_BYTES_ARCHIVED_NO_VERDICT_CHANGED')
