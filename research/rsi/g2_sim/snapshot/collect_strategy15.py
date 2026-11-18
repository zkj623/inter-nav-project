from pathlib import Path
from collections import defaultdict,Counter
import json,zipfile,datetime,hashlib
R=Path(__file__).resolve().parent
catalog=json.loads((R/'rigid_catalog.json').read_text()); rows=[]
variants=['','retry','surface','tilt','clear','frame','chair','chairfar','horizontal','edge','adaptive','curve','guard','level','balance','feedback']
for obj in catalog['objects']:
 aid=obj['asset'];trials=[];planning=[]
 for v in variants:
  prefix='strategy15'+('_'+v if v else '')
  for p in (R/(prefix+'_trials')).glob(f'*/{aid}/*/result.json'):
   d=json.loads(p.read_text());pd=p.parent.parent.parent.name;planfile=R/pd/(aid+'.json');plan_doc=json.loads(planfile.read_text()) if planfile.exists() else {};plans=plan_doc.get('plans',[]);c=d.get('candidate',0);plan=plans[c] if c<len(plans) else {}
   keys=['asset','candidate','repeat','success','failure','error','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_relative_drift_m','hold_four_finger_contact_fraction','max_contact_surface_error_m','source_sha256','mass_kg']
   tr={k:d.get(k) for k in keys};tr['valid_physical_result']=d.get('failure') not in ('HARNESS_ERROR','NUMERICAL_INSTABILITY') and not (p.parent/'cancellation.json').exists();source=R/('grasp_trial_strategy15'+('_'+v if v else '')+'.py');tr['current_source_matches_executed_hash']=source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==d.get('source_sha256');tr.update(variant=v or 'primary',path=str(p.relative_to(R)),strategy=plan.get('strategy'),mode=plan.get('mode'),plan_path=str(planfile.relative_to(R)));trials.append(tr)
  p=R/(prefix+'_validated_plans')/(aid+'.json')
  if p.exists():
   d=json.loads(p.read_text());planning.append(dict(variant=v or 'primary',plans=len(d.get('plans',[])),rejections={k:val for k,val in d.items() if 'reject' in k}))
 groups=defaultdict(list)
 for t in trials:groups[(t['variant'],t['candidate'])].append(t)
 successes=[k for k,ts in groups.items() if len([t for t in ts if t['success']])>=3]
 passes=[t for t in trials if t['success']]
 best=max(trials,key=lambda t:(bool(t['success']),max(0.,t.get('min_hold_lift_m') or 0.) if (t.get('min_hold_lift_m') or 0.)>.05 else 0.,t.get('max_lift_m') or 0)) if trials else None
 rows.append(dict(asset=aid,name=obj.get('name',aid),status='repeat_pass' if successes else 'single_pass' if passes else 'tested_not_passed' if trials else 'planned_not_tested' if any(x['plans'] for x in planning) else 'no_admissible_plan',repeat_pass_groups=[list(x) for x in successes],trial_count=sum(t['valid_physical_result'] for t in trials),invalid_or_cancelled_count=sum(not t['valid_physical_result'] for t in trials),passes=len(passes),best=best,trials=trials,planning=planning))
out=dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=dict(Counter(r['status'] for r in rows)),objects=rows,assumptions=['1 kg rigid bodies','ideal torso drives','no attachments or object pose tracking','three identical initial-condition repeats are reproducibility, not robustness','lift and 3 s hold only; no loaded body rise/navigation'])
(R/'strategy15_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out['counts']))
for r in rows:
 b=r['best'] or {};print(r['asset'],r['status'],r['trial_count'],b.get('variant'),b.get('min_hold_lift_m'),b.get('failure'))
manifest=[]
for folder in ['object_assets','strategy15_surface_assets','strategy15_surface_proxies','robot_assets']:
 for p in (R/folder).glob('*'):
  if not p.is_file():continue
  if folder=='robot_assets' and p.name!='g2_contact_inertia_fixed.usdc':continue
  if p.suffix not in ('.usdc','.npz','.json'):continue
  manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'strategy15_asset_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(R/'strategy15_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.write(R/'strategy15_asset_manifest.json','asset_manifest.json')
 for p in (R/'strategy15_surface_proxies').glob('*'):
  if p.suffix in ('.npz','.json'):z.write(p,str(p.relative_to(R)))
 z.write(R/'strategy15_summary.json','summary.json');z.write(R/'rigid_catalog.json','rigid_catalog.json')
 for p in R.glob('*strategy15*.py'):z.write(p,'source/'+p.name)
 for p in R.glob('strategy15*_plans/*.json'):z.write(p,str(p.relative_to(R)))
 for prefix in ['strategy15',*[f'strategy15_{v}' for v in variants if v]]:
  for p in (R/(prefix+'_trials')).glob('*/*/*/*.json'):z.write(p,str(p.relative_to(R)))
