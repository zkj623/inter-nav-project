from pathlib import Path
from collections import defaultdict,Counter
import json
R=Path(__file__).resolve().parent;out=defaultdict(Counter)
for p in R.glob('collection_trials/**/result.json'):
 d=json.loads(p.read_text());out[d['asset']]['HARNESS_PASS' if d['success'] else d.get('failure','unknown')]+=1
for p in R.glob('collection_trials/**/verification.json'):
 d=json.loads(p.read_text())
 if d['harness_success'] and not d['verified_success']:out[d['asset']]['AUDIT:'+str(d.get('rejections'))]+=1
print(json.dumps(out,indent=2))
