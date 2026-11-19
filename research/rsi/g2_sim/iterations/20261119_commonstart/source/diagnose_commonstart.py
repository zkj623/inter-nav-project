from pathlib import Path
import json
R=Path(__file__).resolve().parent
for f in sorted(R.glob('commonstart_trials/**/result.json')):
 d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());print(d['asset'],d['success'],d.get('failure'),flush=True)
 for s in ss[-2:]:
  worst=sorted([(n,abs(v-s['joint_targets'].get(n,0))) for n,v in s['joint_positions'].items() if n in s['joint_targets']],key=lambda x:-x[1])[:4]
  print(s['phase'],s['t'],s['max_tracking_error_rad'],worst,flush=True)
