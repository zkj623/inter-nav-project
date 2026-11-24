from pathlib import Path
import json,time,subprocess,sys
W=Path(__file__).resolve().parent
# Keep held-out results separate from deliberate training-position overfit tests.
while True:
 matches=list((W/'evaluations').glob('*expanded*20000/summary.json'))
 complete=[p for p in matches if json.loads(p.read_text()).get('total',0)>=6]
 if complete:break
 time.sleep(30)
rows=json.loads((W/'datasets/expanded12/manifest.json').read_text());assets=list(dict.fromkeys(r['asset'] for r in rows if r['split']=='validation'))
if not assets:raise RuntimeError('NO_HELDOUT_EPISODES')
checkpoint=W/'runs/act_expanded12/step_020000.pt'
with (W/'act_heldout_eval.log').open('w') as log:subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint',str(checkpoint),'--run','act_expanded12_step020000_heldout','--dataset','expanded12','--split','validation','--assets',*assets,'--evaluation-label','Held-out object placements; closed-loop learned policy; no expert feedback'],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
print('ACT_HELDOUT_EVALUATION_FINISHED',flush=True)
