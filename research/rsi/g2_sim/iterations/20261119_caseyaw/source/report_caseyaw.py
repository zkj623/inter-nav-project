from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
for f in sorted(R.glob('caseyaw*_trials/**/result.json')):
 d=json.loads(f.read_text());print(str(f.relative_to(R)),{k:d.get(k) for k in ['success','failure','max_lift_m','min_hold_lift_m','hold_four_finger_contact_fraction','diagnostic_four_available_fraction']},flush=True)
for f in sorted(R.glob('caseyaw*_trials/**/progress.json')):
 if f.with_name('result.json').exists():continue
 d=json.loads(f.read_text());print('ACTIVE',str(f.relative_to(R)),d['phase'],d['t'],d['lift_m'],d['tilt_deg'],flush=True)
diagnostics=[]
for f in R.glob('caseyaw_diag_trials/**/result.json'):
 d=json.loads(f.read_text());row={k:d.get(k) for k in ['asset','success','failure','diagnostic_only','diagnostic_note','diagnostic_four_available_fraction','hold_four_finger_contact_fraction','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','hold_relative_drift_m','source_sha256']};row['result']=str(f.relative_to(R));row['result_mtime_utc']=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat()
 for name in ['samples.json','review.mp4']:
  p=f.with_name(name);row[name]=dict(server_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 row['diagnostic_only']=True
 diagnostics.append(row)
(R/'caseyaw_diagnostic_index.json').write_text(json.dumps(diagnostics,indent=2))
