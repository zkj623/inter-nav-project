from pathlib import Path
import json,datetime,re
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
out=dict(time=datetime.datetime.now().isoformat(),data={},training={},evaluations=[],live=[])
for directory in ['collection_state','wide_collection_state']:
 counts={p.stem:len(json.loads(p.read_text()).get('accepted',[])) for p in (R/directory).glob('*.json')};out['data'][directory]=dict(total=sum(counts.values()),per_asset=counts)
for run in ['act_pilot4','act_pilot4_refine','act_pilot4_robust','act_expanded12']:
 p=W/'runs'/run/'metrics.jsonl'
 if p.exists():
  metrics=[json.loads(line) for line in p.read_text().splitlines() if line.endswith('}')];out['training'][run]=dict(step=max(m['step'] for m in metrics),error=next((m for m in reversed(metrics) if 'split' in m),None))
for log in ['pi05_train.log','pi05_relative_train.log']:
 p=W/log
 if p.exists():
  text=p.read_text(errors='replace');steps=re.findall(r'\[step=(\d+)\]',text);out['training'][log]=dict(last_checkpoint_step=int(steps[-1]) if steps else None,last_lines=text.splitlines()[-2:])
for p in (W/'evaluations').glob('*/summary.json'):
 s=json.loads(p.read_text());out['evaluations'].append(dict(run=s['run'],successes=s['successes'],total=s['total'],results=[dict(asset=r['asset'],success=r['success'],failure=r['failure'],lift_m=r.get('lift_m')) for r in s['results']]))
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=proc.read_bytes().split(b'\0');args=[x.decode() for x in args if x]
  if len(args)>1 and str(R/'policy_') in args[1]:
   def val(flag):return args[args.index(flag)+1]
   trial=R/'policy_trials'/val('--run')/val('--plan-dir')/val('--asset')/'candidate0_repeat1';p=trial/'live.json'
   if p.exists():out['live'].append(dict(pid=int(proc.parent.name),run=val('--run'),asset=val('--asset'),**json.loads(p.read_text())))
 except (OSError,ValueError,json.JSONDecodeError):pass
print(json.dumps(out,ensure_ascii=False));(W/'brief_status.json').write_text(json.dumps(out,indent=2))

