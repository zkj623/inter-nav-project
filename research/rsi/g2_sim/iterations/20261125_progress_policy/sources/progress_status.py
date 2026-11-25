from pathlib import Path
import json,time,sys
W=Path(__file__).resolve().parent
def status():
 out={'time':time.strftime('%FT%T'),'runs':{}}
 for run,folder in [('act_progress_v1','progress_act_20261125'),('act_progress_v1_execute1','receding_act_20261125'),('act_timeonly_v2','timeonly_act_20261125'),('act_force_v1','force_act_20261125'),('act_dagger_v1','dagger_act_20261125')]:
  r={};p=W/'runs'/run/'metrics.jsonl'
  if p.exists():r['training']=json.loads(p.read_text().splitlines()[-1])
  p=W/'evaluations'/(run if run.endswith('execute1') else run+'_8000')/'summary.json'
  if p.exists():
   s=json.loads(p.read_text());r['evaluation']={k:s[k] for k in ['successes','total']};r['evaluation']['results']=[{k:x.get(k) for k in ['asset','success','failure','lift_m']} for x in s['results']]
  for name in ['stopped.json','time_baseline_finished.json','early_stop.json','finished.json','needs_data_iteration.json']:
   p=W/folder/name
   if p.exists():r[name]=json.loads(p.read_text())
  p=W/folder/'resources.jsonl'
  if p.exists():
   s=json.loads(p.read_text().splitlines()[-1]);r['resources']={k:s[k] for k in ['time','available_gib','swap_io_mib_s','iowait_percent','load']}
  if r:out['runs'][run]=r
 print(json.dumps(out),flush=True)
status()
if '--watch' in sys.argv:
 for _ in range(240):time.sleep(50);status()
