from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in R.glob('commonfix_solver*_trials/**/result.json'):
 d=json.loads(f.read_text())
 if d['asset']!='case_handheld':continue
 nc=json.loads(f.with_name('nonhand_contacts.json').read_text());hits=sorted([x for x in nc if x['impulse']>.001],key=lambda x:-x['impulse'])
 print(str(f.relative_to(R)),d.get('failure'),hits[:2],flush=True)
