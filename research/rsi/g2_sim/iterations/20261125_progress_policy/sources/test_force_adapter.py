from pathlib import Path
import json,sys,tempfile,subprocess
from unittest.mock import patch
import h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';sys.path.insert(0,str(R))
from policy_force_recorder import Recorder
from policy_time_recorder import Recorder as TimeRecorder
from policy_threecam_recorder import Recorder as NetworkRecorder
from g2_force_features import FINGER_NAMES
samples=[dict(physics_time_s=t,finger_forces=dict.fromkeys(FINGER_NAMES,v)) for t,v in [(1.,1.),(1.1,2.),(1.2,6.),(1.3,100.)]]
with tempfile.TemporaryDirectory(prefix='g2-force-adapter-') as directory:
 t=Path(directory);r=Recorder(t,['q']*16,dict(asset='test'));r.count=8;r.set_contact_observation(samples,1.25)
 with patch.object(NetworkRecorder,'post',autospec=True,return_value={}) as post:
  r.post('/predict',dict(qpos=[0.]*16))
  request=post.call_args.args[2];assert request['elapsed_s']==.35 and request['finger_forces']==[3.]*4
 r.count=0;r.file=h5py.File(t/'episode.hdf5','w');r.file['timestamps']=[1.25]
 def capture(self,*args,**kwargs):self.count+=1;return {'held':True}
 with patch.object(TimeRecorder,'command',capture):assert r.command(None)=={'held':True}
 assert np.array_equal(r.file['observations/finger_forces'][0],[3.]*4)
 assert r.file['force_timestamps'][0]==1.2;r.file.close()
 (t/'samples.json').write_text(json.dumps(samples))
 subprocess.run([sys.executable,str(W/'audit_force_episode.py'),'--trial',str(t)],check=True)
 with h5py.File(t/'episode.hdf5','a') as f:f['force_timestamps'][0]=1.3
 bad=subprocess.run([sys.executable,str(W/'audit_force_episode.py'),'--trial',str(t)],capture_output=True)
 assert bad.returncode!=0,'Future timestamp accepted'
result=dict(force_and_elapsed_sent_together=True,recorded_values_match_network=True,independent_causal_audit_passed=True,tampered_future_timestamp_rejected=True)
(W/'force_adapter_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
