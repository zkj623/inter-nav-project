from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
for f in R.glob('symhome_v2_trials/**/samples.json'):
 ss=json.loads(f.read_text());p=[x for x in ss if x['phase']=='prepare_home'];worst={}
 for x in p:
  for n,t in x['target_joint_positions'].items():worst[n]=max(worst.get(n,0.),abs(x['joint_positions'].get(n,0.)-t))
 print(f.parts[-3],sorted(worst.items(),key=lambda x:-x[1])[:6],flush=True)
