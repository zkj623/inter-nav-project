from pathlib import Path
import json,statistics,time,html
W=Path(__file__).resolve().parent;O=W/'pi05_focus_20261125'
metrics=[json.loads(line) for line in (O/'pilot8_relative_focus_v1_metrics.jsonl').read_text().splitlines() if line.strip()]
stages=[]
for path in sorted((W/'evaluations').glob('pi05_focus*relative*/summary.json')):
 d=json.loads(path.read_text());step=int(Path(d['checkpoint']).name)+1
 if d.get('dataset_split')!='train':continue
 items=[]
 for r in d['results']:
  trial=Path(r['trial']);p=trial/'samples.json';samples=json.loads(p.read_text()) if p.exists() else []
  longest=span=0.;previous=None;high=[]
  for s in samples:
   if s['lift_m']>=.15:
    if previous is not None and 0<s['t']-previous<=.15:span+=s['t']-previous
    else:span=0.
    longest=max(longest,span);previous=s['t'];high.append(s)
   else:span=0.;previous=None
  p=trial/'nonhand_contacts.json';contacts=json.loads(p.read_text()) if p.exists() else []
  record={k:r.get(k) for k in ['asset','success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg']}
  record.update(above15_contiguous_span_s=longest,above15_tilt_range_deg=[min(s['tilt_deg'] for s in high),max(s['tilt_deg'] for s in high)] if high else None,nonhand_contact_links=sorted({c.get('actor','unknown') for c in contacts}),video=str(O/'reviews'/d['run']/(r['asset']+'.mp4')))
  items.append(record)
 recent=[r['loss'] for r in metrics if step-300<=r['step']<step]
 stages.append(dict(protocol=d.get('protocol','stable'),run=d['run'],steps=step,equivalent_epochs=step*2/8595,sampling_seed=d.get('sampling_seed',(d['results'][0].get('policy') or {}).get('sampling_seed') if d['results'] else None),successes=d['successes'],total=d['total'],loss_last300_mean=statistics.mean(recent) if recent else None,items=items))
stages.sort(key=lambda x:(x['steps'],x['run']))
report=dict(updated=time.strftime('%FT%T'),training_frames=8595,batch_size=2,stages=stages,notes='One rollout per object per seed. Protocols are listed per stage; success rates under different protocols are not directly comparable. Earlier 6k/18k/36k baselines use differing seeds; long stages share seed 20261125. Successful 4/4 stages get seed 20261126 repetition. Lift-height spans do not alone imply stable success.')
tmp=O/'longitudinal_report.tmp.json';tmp.write_text(json.dumps(report,indent=2));tmp.replace(O/'longitudinal_report.json')
parts=['<!doctype html><meta charset="utf-8"><title>G2 pi05 training comparison</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>','<h1>G2 pi05 training and closed-loop comparison</h1>',f'<p>Updated {report["updated"]}</p>',f'<p>{html.escape(report["notes"])}</p>','<table><tr><th>Updates / epochs</th><th>Seed</th><th>Stable lifts</th><th>Mean loss</th><th>Videos</th></tr>']
for st in stages:
 parts.append(f'<tr><td>{st["steps"]} / {st["equivalent_epochs"]:.2f}</td><td>{st["sampling_seed"]}</td><td>{st["successes"]}/{st["total"]}</td><td>{st["loss_last300_mean"]}</td><td><a href="reviews/{html.escape(st["run"])}/index.html">Review</a></td></tr>')
parts.append('</table>')
for st in stages:
 parts.append('<p>Protocol: '+html.escape(st['protocol'])+'</p>')
 parts.append(f'<h2>{html.escape(st["run"])}</h2><table><tr><th>Object</th><th>Max lift cm</th><th>Continuous above 15 cm seconds</th><th>Outcome</th></tr>')
 for r in st['items']:
  parts.append(f'<tr><td>{html.escape(r["asset"])}</td><td>{100*(r["max_lift_m"] or 0):.2f}</td><td>{r["above15_contiguous_span_s"]:.2f}</td><td>{html.escape("PASS" if r["success"] else str(r["failure"]))}</td></tr>')
 parts.append('</table>')
(O/'longitudinal_report.html').write_text('\n'.join(parts));print('COMPARISON_READY',str(O/'longitudinal_report.json'),'stages',len(stages))
