"""Checked velocity feedback for persistent holds after forward travel."""
import numpy as np
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


class ForwardStopFeedback:
    """Keep a learned velocity policy near zero until the next motion intent.

    Only an explicit stop after predominantly forward travel can arm this
    controller. Lateral/yaw-only stops retain their separately calibrated
    behavior. Feedback is NOT released at a transient base-velocity minimum.
    Every modified command must pass the caller's current-map motion checker.
    """
    def __init__(self, gain=1.5, linear_limit=.4, yaw_limit=.6):
        self.gain=float(gain);self.linear_limit=float(linear_limit);self.yaw_limit=float(yaw_limit)
        self.limiter=VelocitySlewLimiter();self.active=False;self.eligible=False;self.rejoining=False
        self.last_detail={}

    def apply(self,requested,measured,now,*,stop_requested=False,feedback_allowed=True,healthy=True,checker=None):
        requested=np.asarray(requested,float).reshape(3);measured=np.asarray(measured,float).reshape(3)
        if not np.isfinite(requested).all() or not np.isfinite(measured).all():
            raise ValueError('Stop feedback requires finite command and measured velocity')
        self.limiter.observe(measured,now)
        if not healthy:
            self.active=False;self.eligible=False;self.rejoining=False
            applied=np.zeros(3);self.limiter.accept(applied,now)
            self.last_detail={'mode':'emergency_zero','active':False,'motion_safe':None}
            return applied
        moving=bool(np.any(np.abs(requested)>1e-9))
        if not feedback_allowed:
            self.rejoining=self.rejoining or self.active;self.active=False;self.eligible=False
        elif stop_requested:
            self.active=self.active or self.eligible
        elif moving:
            self.rejoining=self.rejoining or self.active;self.active=False
            forward=bool(requested[0]>0 and abs(requested[1])<=.5*requested[0] and abs(requested[2])<=.35)
            if forward and requested[0]>=.20:self.eligible=True
            elif not forward:self.eligible=False
            # Preserve forward eligibility while slowing down, but do not
            # arm from tiny cross-axis residues of a lateral/yaw maneuver.
        # A zero command without a new intent preserves an already active hold.
        target=requested.copy()
        if self.active:
            target=-self.gain*self.limiter.measured
            speed=float(np.linalg.norm(target[:2]))
            if speed>self.linear_limit:target[:2]*=self.linear_limit/speed
            target[2]=np.clip(target[2],-self.yaw_limit,self.yaw_limit)
        modified=self.active or self.rejoining
        safe=True
        if modified:
            if checker is None:raise ValueError('A current-map checker is required for feedback motion')
            applied,safe=checker(target,self.limiter,now)
            applied=np.asarray(applied,float) if safe else np.zeros(3)
            if self.rejoining and np.allclose(applied,requested,atol=1e-9):self.rejoining=False
        else:applied=requested.copy()
        self.limiter.accept(applied,now)
        self.last_detail={'mode':'forward_hold' if self.active else ('rejoining' if modified else 'unchanged'),
            'active':self.active,'eligible':self.eligible,'motion_safe':bool(safe),
            'requested':requested.tolist(),'feedback_target':target.tolist(),'applied':applied.tolist()}
        return applied
