"""Measured recovery stops remain latched through changing map classifications."""
import numpy as np
from grutopia_extension.interactive_navigation.go2_interaction_motion import MeasuredStopGate


class RecoveryMotionGate:
    def __init__(self):
        self.mode=None;self.direction=None;self.settling=False;self.stopped=False
        self.stop=MeasuredStopGate(hold_seconds=.20)
        self.instant=MeasuredStopGate(hold_seconds=0.)

    @staticmethod
    def quiet(measured):
        return np.linalg.norm(measured[:2])<.06 and abs(measured[2])<.10

    def _state_quiet(self,measured,now,body_state,command,gate):
        command=np.zeros(3) if command is None else np.asarray(command,float)
        if (not self.quiet(measured) or not np.isfinite(command).all()
                or np.linalg.norm(command[:2])>=.03 or abs(command[2])>=.06):
            gate.reset();return False
        if body_state is None:
            # Legacy static controller tests have only a planar measurement.
            # The live runner always supplies complete observed body motion.
            position=np.zeros(3);linear=np.array([*measured[:2],0.]);angular=np.array([0.,0.,measured[2]]);tilt=0.
        else:
            position,linear,angular,tilt,observed_at=body_state
            if (abs(now-observed_at)>.10 or any(np.asarray(v).shape!=(3,) for v in [position,linear,angular])):
                gate.reset();return False
        return gate.update(position,linear,angular,tilt,now)

    def _quiet_now(self,measured,now,body_state,command):
        self.instant.reset()
        return self._state_quiet(measured,now,body_state,command,self.instant)

    def _begin_stop(self):
        if not self.settling:self.stop.reset()
        self.settling=True;self.stopped=False

    def _finish_stop(self):
        self.mode=None;self.direction=None;self.settling=False;self.stopped=True

    def filter(self,request,measured,now,*,body_state=None,command=None):
        request=np.asarray(request,float);measured=np.asarray(measured,float)
        length=np.linalg.norm(request[:2])
        if length<1e-8 and abs(request[2])<1e-8:
            if self.mode is not None or self.settling or not self._quiet_now(measured,now,body_state,command):
                self._begin_stop()
                if self._state_quiet(measured,now,body_state,command,self.stop):self._finish_stop()
            return request
        mode='turn' if length<1e-8 else 'translation'
        direction=request[:2]/length if length>=1e-8 else None
        reversal=(mode==self.mode=='translation' and self.direction is not None
                  and float(np.dot(direction,self.direction))<.5)
        changed=self.mode is not None and (mode!=self.mode or reversal)
        if changed or (self.mode is None and not self._quiet_now(measured,now,body_state,command)):
            self._begin_stop()
        # A pending leave_recovery stop must survive a subsequent blocked
        # map classification, even when the proposed motion has the same mode.
        if self.settling:
            if not self._state_quiet(measured,now,body_state,command,self.stop):
                self.mode=mode;self.direction=direction
                return np.zeros(3)
            self._finish_stop()
        self.mode=mode;self.direction=direction;self.stopped=False
        return request

    def leave_recovery(self,measured,now,*,body_state=None,command=None):
        measured=np.asarray(measured,float)
        if self.stopped and self._quiet_now(measured,now,body_state,command):return True
        self._begin_stop()
        if not self._state_quiet(measured,now,body_state,command,self.stop):return False
        self._finish_stop();return True
