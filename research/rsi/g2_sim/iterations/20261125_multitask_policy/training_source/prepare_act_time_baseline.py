from pathlib import Path
import json,hashlib,h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';D=W/'datasets/pilot4_one_each';D.mkdir(exist_ok=True)
all_rows=json.loads((W/'datasets/pilot4/manifest.json').read_text());rows=[next(r for r in all_rows if r['asset']==asset) for asset in dict.fromkeys(r['asset'] for r in all_rows)];stats=json.loads((W/'datasets/pilot4/stats.json').read_text());values={'qpos':[],'action':[]}
for row in rows:
 with h5py.File(row['cache']) as f:
  for key in values:values[key].append(f[key][:])
for key,arrays in values.items():
 x=np.concatenate(arrays).astype(np.float64)
 for suffix,v in [('mean',x.mean(0)),('std',np.maximum(x.std(0),.01)),('q01',np.quantile(x,.01,axis=0)),('q99',np.quantile(x,.99,axis=0))]:stats[key+'_'+suffix]=v.tolist()
stats['extra_input']='elapsed control time in seconds divided by10; no expert phase or trajectory inputs'
(D/'stats.json').write_text(json.dumps(stats,indent=2));(D/'manifest.json').write_text(json.dumps(rows,indent=2));(D/'scope.json').write_text(json.dumps(dict(episodes=4,assets=[r['asset'] for r in rows],purpose='Explicit training-position overfit baseline; no generalization claim'),indent=2))
adapter='''from pathlib import Path
import hashlib
from policy_threecam_recorder import Recorder as Base
class Recorder(Base):
 def post(self,path,value):
  if path=='/predict':value={**value,'elapsed_s':(self.count-1)/20.}
  return super().post(path,value)
 def configure_policy(self,*args,**kwargs):
  super().configure_policy(*args,**kwargs)
  self.policy_info['base_adapter_sha256']=self.adapter_sha256
  self.adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  self.policy_info['extra_input']='elapsed task time in seconds; no expert phase, targets or trajectory retrieval'
'''
(R/'policy_time_recorder.py').write_text(adapter)
for row in rows:
 asset=row['asset'];p=R/f'policy_time_{asset}_trial.py';s=(R/f'policy_{asset}_trial.py').read_text().replace('from policy_threecam_recorder import','from policy_time_recorder import');p.write_text(s);(R/'source_versions'/(hashlib.sha256(p.read_bytes()).hexdigest()+'.py')).write_bytes(p.read_bytes())
print('ACT_TIME_BASELINE_READY',len(rows))
