from pathlib import Path
import json,subprocess,sys,concurrent.futures
R=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (R/'dataset/manifest.jsonl').read_text().splitlines()]
todo=[]
for row in rows:
 D=Path(row['raw_trial']);v=json.loads((D/'verification.json').read_text())
 if v.get('audit_version',0)<2:todo.append(D)
print('REQUIRING_CURRENT_AUDIT',len(todo),[str(x.relative_to(R)) for x in todo],flush=True)
def audit(D):
 p=subprocess.run([sys.executable,str(R/'verify_collection.py'),'--trial-dir',str(D)],cwd=R,capture_output=True,text=True)
 (D/'final_reaudit.log').write_text(p.stdout+p.stderr)
 return dict(trial=str(D),returncode=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(audit,todo):print(json.dumps(r),flush=True)
