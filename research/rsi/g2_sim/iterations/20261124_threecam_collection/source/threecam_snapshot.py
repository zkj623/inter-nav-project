from pathlib import Path
import json,time
R=Path(__file__).resolve().parent
if (R/'collection_progress.json').exists():print(json.dumps(json.loads((R/'collection_progress.json').read_text())),flush=True)
else:
 for i,n in [(1,2),(2,2),(3,1),(4,1)]:
  row={'placement':i,'repeat':n}
  for root in ['realpose_trials','threecam_replay_trials']:
   T=R/root/f'placements/p{i:04d}/box_white/candidate0_repeat{n}';sub={}
   for name in ['live','result','acceptance']:
    f=T/(name+'.json')
    if f.exists():
     d=json.loads(f.read_text());sub[name]={k:d[k] for k in ['t','phase','lift_m','success','failure','accepted','rejections'] if k in d}
   row[root]=sub
  print(json.dumps(row),flush=True)
if (R/'threecam_pilot_verified.json').exists():print('PILOT_GATE',json.loads((R/'threecam_pilot_verified.json').read_text()),flush=True)
