from pathlib import Path
import json,hashlib,numpy as np
from policy_force_recorder import Recorder as ForceRecorder
from threecam_recorder_v3 import Recorder as ThreeCameraRecorder
class Recorder(ForceRecorder):
 teacher_mode=False
 teacher_start_frame=None
 def start_teacher(self):
  self.teacher_mode=True;self.teacher_start_frame=self.count;self.action_queue=[]
 def command(self,*args,**kwargs):
  previous=self.count
  if self.teacher_mode:
   held=ThreeCameraRecorder.command(self,*args,**kwargs)
   if self.count>previous:
    for name,val in [('observations/finger_forces',self.contact_force),('force_timestamps',self.contact_stamp)]:
     self.file[name].resize(self.count,axis=0);self.file[name][self.count-1]=val
    assert abs(float(self.file['timestamps'][self.count-1])-self.contact_now)<1e-8
  else:held=super().command(*args,**kwargs)
  if self.count>previous:
   if 'behavior_source' not in self.file:self.file.create_dataset('behavior_source',shape=(0,),maxshape=(None,),dtype='u1')
   self.file['behavior_source'].resize(self.count,axis=0);self.file['behavior_source'][self.count-1]=int(self.teacher_mode)
  return held
 def finish(self,result):
  if self.file is not None:
   self.file.attrs['collection_role']='DAgger policy prefix plus expert corrective suffix; train on suffix only'
   self.file.attrs['teacher_start_frame']=self.teacher_start_frame if self.teacher_start_frame is not None else -1
  if hasattr(self,'policy_info'):
   self.policy_info={**self.policy_info,'expert_feedback':True,'model':'DAGGER_EXPERT_RECOVERY_NOT_POLICY_EVALUATION'}
  super().finish(result)
  result['collection_role']='DAGGER_EXPERT_RECOVERY_NOT_POLICY_EVALUATION'
  result['teacher_start_frame']=self.teacher_start_frame
  result['policy_evaluation']['control_scope']='Learned ACT prefix, then expert contact/IK correction; never count as pure policy success'
  result['recovery_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  path=self.out/'metadata.json';m=json.loads(path.read_text());m['collection_role']=result['collection_role'];m['teacher_start_frame']=self.teacher_start_frame;m['training_frame_range']=[self.teacher_start_frame,self.count];path.write_text(json.dumps(m,indent=2))
