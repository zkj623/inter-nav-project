"""Continue learned feedback after the final timed observation window."""
from pathlib import Path
import hashlib
from policy_threecam_recorder import Recorder as PolicyRecorder
class Recorder(PolicyRecorder):
 def configure_policy(self,*args,**kwargs):
  super().configure_policy(*args,**kwargs)
  self.hold_info=None
 def mark_hold(self):
  if self.hold_info is None:
   self.hold_info=dict(frame=self.count,inference_calls_at_trigger=len(self.predict_log),trigger='final 10.1 seconds of fixed per-object rollout; time label only',queued_actions_at_trigger=len(self.action_queue))
 def finish(self,result):
  if self.hold_info is not None:
   self.hold_info['inference_calls_after_trigger']=len(self.predict_log)-self.hold_info['inference_calls_at_trigger']
   self.hold_info['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  result['policy_continuation']=self.hold_info
  super().finish(result)
