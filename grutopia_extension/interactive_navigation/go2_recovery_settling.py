"""Measured settling at recovery-mode changes, without gating every gait tick."""
import numpy as np


class RecoveryMotionGate:
    def __init__(self):
        self.mode=None;self.direction=None;self.settling=False
        self.stable_since=None;self.exit_stable_since=None

    @staticmethod
    def quiet(measured):
        return np.linalg.norm(measured[:2])<.06 and abs(measured[2])<.10

    def filter(self,request,measured,now):
        request=np.asarray(request,float);measured=np.asarray(measured,float)
        self.exit_stable_since=None
        length=np.linalg.norm(request[:2])
        if length<1e-8 and abs(request[2])<1e-8:return request
        mode='turn' if length<1e-8 else 'translation'
        direction=request[:2]/length if length>=1e-8 else None
        reversal=(mode==self.mode=='translation' and self.direction is not None
                  and float(np.dot(direction,self.direction))<.5)
        changed=self.mode is not None and (mode!=self.mode or reversal)
        if changed or (self.mode is None and not self.quiet(measured)):
            self.settling=True;self.stable_since=None
        self.mode=mode;self.direction=direction
        if self.settling:
            if self.quiet(measured):
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.20:self.settling=False
            else:self.stable_since=None
        return np.zeros(3) if self.settling else request

    def leave_recovery(self,measured,now):
        if not self.quiet(measured):
            self.exit_stable_since=None
            return False
        if self.exit_stable_since is None:self.exit_stable_since=now
        if now-self.exit_stable_since<.20:return False
        self.mode=None;self.direction=None;self.settling=False;self.stable_since=None
        return True
