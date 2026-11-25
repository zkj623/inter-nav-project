from pathlib import Path
import json,re
W=Path(__file__).resolve().parent;O=W/'pi05_focus_20261125'
def tail(p,limit=65536):
 if not p.exists():return ''
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-limit));return f.read().decode(errors='replace')
status=json.loads((O/'status.json').read_text());out={'status':status}
if status['phase']=='training':
 log=O/(status['run']+'_train_'+str(status['target_steps'])+'.log');s=tail(log)
 matches=re.findall(r'Progress on: ([\d.]+)(k?)it/.*?rate:([^ ]+) remaining:([^ ]+)',s)
 if matches:
  n,k,rate,eta=matches[-1];out['approximate_current_step']=round(float(n)*(1000 if k else 1));out['rate']=rate;out['log_eta']=eta
 proc=Path('/proc')/str(status.get('pid',0))/'cmdline';out['trainer_alive']=proc.exists() and b'train_pi05_focus' in proc.read_bytes()
 out['recent_errors']=[x for x in s.splitlines() if any(k in x for k in ['Traceback','Error:','RESOURCE_EXHAUSTED'])][-4:]
metrics=tail(O/'resources.jsonl').splitlines()
if metrics:out['resources']=json.loads(metrics[-1])
results=[]
for p in sorted((W/'evaluations').glob('pi05_focus*/summary.json')):
 d=json.loads(p.read_text());results.append(dict(run=p.parent.name,successes=d['successes'],total=d['total'],items=[{k:r.get(k) for k in ['asset','success','failure','rejections','max_lift_m','min_hold_lift_m']} for r in d['results']]))
out['evaluations']=results
print(json.dumps(out,indent=2))
