from pathlib import Path
import hashlib,numpy as np
from policy_time_recorder import Recorder as Base
from g2_force_features import causal_force
class Recorder(Base):
 def set_contact_observation(self,samples,now):
  self.contact_force,self.contact_stamp=causal_force(samples,now)
  self.contact_now=now
 def post(self,path,value):
  if path=='/predict':value={**value,'finger_forces':self.contact_force.tolist()}
  return super().post(path,value)
 def command(self,*args,**kwargs):
  previous=self.count;held=super().command(*args,**kwargs)
  if self.count>previous:
   for name,val in [('observations/finger_forces',self.contact_force),('force_timestamps',self.contact_stamp)]:
    val=np.asarray(val);f=self.file
    if name not in f:f.create_dataset(name,shape=(0,*val.shape),maxshape=(None,*val.shape),dtype='f8')
    f[name].resize(self.count,axis=0);f[name][self.count-1]=val
   assert abs(float(self.file['timestamps'][self.count-1])-self.contact_now)<1e-8
  return held
 def finish(self,result):
  super().finish(result)
  result['policy_evaluation']['force_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  result['policy_evaluation']['force_features_sha256']=hashlib.sha256((Path(__file__).parent/'g2_force_features.py').read_bytes()).hexdigest()
  result['policy_evaluation']['control_scope']='Learned ACT: RGB, qpos, task, elapsed time, latest three past simulated distal finger contact norms; original joint limits, no expert targets or scripted force-balancing controller'
