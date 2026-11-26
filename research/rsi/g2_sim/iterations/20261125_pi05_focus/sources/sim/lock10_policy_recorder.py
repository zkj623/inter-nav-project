"""Stop inference and hold actuator targets, without attaching the object."""
from pathlib import Path
import hashlib
from policy_threecam_recorder import Recorder as PolicyRecorder
from threecam_recorder_v3 import Recorder as ThreeCameraRecorder
class Recorder(PolicyRecorder):
 def configure_policy(self,*args,**kwargs):
  super().configure_policy(*args,**kwargs)
  self.locked_targets=None;self.lock_info=None
 def lock_policy(self):
  if self.locked_targets is not None: return
  self.locked_targets=self.held.copy()
  self.action_queue.clear()
  self.lock_info=dict(frame=self.count,inference_calls_at_lock=len(self.predict_log),targets=self.locked_targets.copy(),trigger='simulator-measured lift >= 0.10m; supervisory stop, not a policy input')
 def command(self,target,phase,robot,world):
  if self.locked_targets is not None:
   return ThreeCameraRecorder.command(self,self.locked_targets,phase,robot,world)
  return super().command(target,phase,robot,world)
 def finish(self,result):
  result['policy_lock']=self.lock_info
  if self.lock_info is not None:
   self.lock_info['inference_calls_after_lock']=len(self.predict_log)-self.lock_info['inference_calls_at_lock']
   self.lock_info['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  super().finish(result)
