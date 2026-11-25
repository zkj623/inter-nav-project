from pathlib import Path
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
