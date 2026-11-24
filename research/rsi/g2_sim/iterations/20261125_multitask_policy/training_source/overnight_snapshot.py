from pathlib import Path
import json,datetime,subprocess
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';out={'time':datetime.datetime.now().isoformat()}
for label,folder in [('original','collection_state'),('wider','wide_collection_state')]:
 counts={};failures={}
 for f in (R/folder).glob('*.json'):
  s=json.loads(f.read_text());counts[s['asset']]=len(s['accepted'])
  for trial in s['attempts']:
   if not trial.get('accepted'):
    key=str(trial.get('failure'));failures[key]=failures.get(key,0)+1
 out[label]=dict(total=sum(counts.values()),counts=counts,failures=failures)
f=W/'runs/act_pilot4/metrics.jsonl'
if f.exists():
 rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()];out['act']=dict(last=rows[-1],zero_latent=[r for r in rows if r.get('split')=='train_zero_latent'][-3:])
out['evaluations']=[json.loads(f.read_text()) for f in (W/'evaluations').glob('*/summary.json')]
out['evaluations']=[{k:v for k,v in r.items() if k!='results'}|{'results':[{'asset':x['asset'],'success':x['success'],'failure':x['failure'],'lift_m':x.get('lift_m')} for x in r['results']]} for r in out['evaluations']]
out['pi05_download_complete']=(W/'weights/pi05_base/DOWNLOAD_COMPLETE').exists()
for label,filename in [('pi05_log','pi05_train.log'),('act_eval_log','act_evaluation_supervisor.log'),('reinforced_nominal','reinforced_nominal.log')]:
 f=W/filename
 if f.exists():out[label]=f.read_text(errors='replace').splitlines()[-5:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines()
(W/'latest_status.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,ensure_ascii=False))
