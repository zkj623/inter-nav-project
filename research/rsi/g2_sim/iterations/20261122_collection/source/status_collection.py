from pathlib import Path
from collections import Counter
import json
R=Path(__file__).resolve().parent;good=Counter();bad=Counter();fail=Counter()
for f in R.glob('collection_plans/*/*_planning.json'):
 d=json.loads(f.read_text());(good if d['success'] else bad)[d['asset']]+=1
 if not d['success']:fail[d.get('reason','').split(' ')[0]]+=1
print(json.dumps(dict(planned=dict(good),rejected=dict(bad),reasons=dict(fail)),indent=2))
