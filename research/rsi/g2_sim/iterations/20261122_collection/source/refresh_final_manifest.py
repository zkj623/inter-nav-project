from pathlib import Path
import json
R=Path(__file__).resolve().parent;p=R/'dataset/manifest.jsonl'
rows=[json.loads(x) for x in p.read_text().splitlines()]
for row in rows:
 v=json.loads((Path(row['raw_trial'])/'verification.json').read_text())
 assert v.get('audit_version',0)>=2 and v['verified_success'], row['raw_trial']
 assert v['episode_sha256']==row['sha256']
 row['verification']=v
p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
print('MANIFEST_CURRENT_AUDIT',len(rows))
