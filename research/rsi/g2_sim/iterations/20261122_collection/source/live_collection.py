from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parent
for p in R.glob('collection_trials/**/live.json'):
 if p.with_name('result.json').exists():continue
 d=json.loads(p.read_text());print(str(p.relative_to(R)),d)
