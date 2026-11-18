from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
def candidates(aid):
 d=json.loads((R/f'push_geometry_{aid}.json').read_text());out=[]
 for mode in ('right_flip','both_flip','original'):
  for old in d['candidates']:
   c=json.loads(json.dumps(old));c['wrist_roll_variant']=mode
   for side,values in c['targets'].items():
    if mode=='original' or (mode=='right_flip' and side!='r'):continue
    t=np.array(values);p=np.array(c['contact_points'][side]);local=t[:3,:3].T@(p-t[:3,3]-np.array([.003,0,0]))
    t[:3,:3]=t[:3,:3]@np.diag([-1.,-1.,1.]);t[:3,3]=p-t[:3,:3]@local-np.array([.003,0,0]);c['targets'][side]=t.tolist()
   if c['mode']=='single_push' and mode=='right_flip':continue
   out.append(c)
 out.sort(key=lambda c:max(np.array(t)[0,3]+c['extent_x']/2 for t in c['targets'].values())+.15*abs(c['contact_height']-.22))
 return out,dict(cached_geometry_candidates=len(d['candidates']),orientation_variants=len(out))
