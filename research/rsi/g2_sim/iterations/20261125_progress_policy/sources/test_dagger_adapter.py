from pathlib import Path
import sys,tempfile,json
from unittest.mock import patch
import h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';sys.path.insert(0,str(R))
from dagger_recovery_recorder import Recorder
from policy_force_recorder import Recorder as Force
from threecam_recorder_v3 import Recorder as Three
with tempfile.TemporaryDirectory() as temp:
 T=Path(temp);r=Recorder(T,['q']*16,dict(asset='test'));r.file=h5py.File(T/'episode.hdf5','w');r.file['timestamps']=[1.,1.05]
 for name,shape in [('observations/finger_forces',(1,4)),('force_timestamps',(1,))]:r.file.create_dataset(name,shape=shape,maxshape=(None,*shape[1:]),dtype='f8')
 r.contact_force=np.array([1.,2.,3.,4.]);r.contact_stamp=1.;r.contact_now=1.05;r.policy_info={};r.action_queue=[1,2]
 def capture(self,*a,**kw):self.count+=1;return {'held':self.count}
 with patch.object(Force,'command',capture):r.command(None)
 r.start_teacher();assert r.teacher_start_frame==1 and r.action_queue==[]
 with patch.object(Force,'command',side_effect=AssertionError('Policy was used after teacher switch')),patch.object(Three,'command',capture):r.command(None)
 assert list(r.file['behavior_source'][:])==[0,1]
 assert np.array_equal(r.file['observations/finger_forces'][1],r.contact_force)
 def finish(self,result):
  result['policy_evaluation']={};(T/'metadata.json').write_text('{}');self.file.close()
 result={}
 with patch.object(Force,'finish',finish):r.finish(result)
 assert r.policy_info['expert_feedback'] is True
 assert result['collection_role']=='DAGGER_EXPERT_RECOVERY_NOT_POLICY_EVALUATION'
 assert json.loads((T/'metadata.json').read_text())['training_frame_range']==[1,2]
out=dict(teacher_switch_skips_policy=True,prefix_labels_excluded=True,contact_inputs_recorded=True,expert_collection_cannot_count_as_policy_success=True)
(W/'dagger_adapter_test.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
