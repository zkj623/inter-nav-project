"""Stop navigation on rising body tilt and require measured settling to resume.

This supervisor requests the original policy's zero-velocity stance. It never
changes a pose, joint state, health threshold, or collision layer.
"""
from collections import deque
import math

import numpy as np


class NavigationAttitudeRecovery:
    def __init__(self, trigger_degrees=18., release_degrees=10.,
                 minimum_hold_ticks=100, stable_ticks=50, maximum_hold_ticks=750):
        if not 0 < release_degrees < trigger_degrees < 30:
            raise ValueError('Recovery must trigger below the unchanged 30 degree failure limit')
        if not 1 < stable_ticks <= minimum_hold_ticks < maximum_hold_ticks:
            raise ValueError('Invalid recovery windows')
        self.trigger_degrees=trigger_degrees;self.release_degrees=release_degrees
        self.minimum_hold_ticks=minimum_hold_ticks;self.maximum_hold_ticks=maximum_hold_ticks
        self.poses=deque(maxlen=stable_ticks)
        self.plateau=deque(maxlen=stable_ticks*2)
        self.active=False;self.failed=False;self.hold_ticks=0
        self.episodes=0;self.total_hold_ticks=0

    def update(self,position,orientation,*,enabled=True):
        p=np.asarray(position,float);q=np.asarray(orientation,float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
            raise ValueError('Invalid observed pose')
        q=q/np.linalg.norm(q)
        tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        started=False
        if not self.active and enabled and tilt>=self.trigger_degrees:
            self.active=True;self.hold_ticks=0;self.poses.clear();self.plateau.clear()
            self.episodes+=1;started=True
        detail={'tilt_degrees':tilt,'started':started,'episode':self.episodes}
        if self.failed:return {'state':'failed',**detail}
        if not self.active:return {'state':'idle',**detail}
        self.hold_ticks+=1;self.total_hold_ticks+=1
        if tilt>self.release_degrees:self.poses.clear()
        else:self.poses.append((p[:2].copy(),yaw))
        stable=False
        if len(self.poses)==self.poses.maxlen:
            origin,heading=self.poses[0]
            travel=max(float(np.linalg.norm(x-origin)) for x,_ in self.poses)
            rotation=max(abs((theta-heading+math.pi)%(2*math.pi)-math.pi) for _,theta in self.poses)
            stable=travel<.025 and rotation<.05
        if tilt>15.:self.plateau.clear()
        else:self.plateau.append((p.copy(),yaw,tilt))
        plateau_stable=False
        if len(self.plateau)==self.plateau.maxlen:
            origin,heading,_=self.plateau[0]
            plateau_stable=(max(float(np.linalg.norm(x[:2]-origin[:2])) for x,_,_ in self.plateau)<.02
                and np.ptp([x[2] for x,_,_ in self.plateau])<.015
                and max(abs((theta-heading+math.pi)%(2*math.pi)-math.pi) for _,theta,_ in self.plateau)<.04
                and np.ptp([a for _,_,a in self.plateau])<1.5)
        stable=stable or plateau_stable
        detail.update(hold_ticks=self.hold_ticks,settled=stable,stable_attitude_plateau=bool(plateau_stable))
        if self.hold_ticks>=self.minimum_hold_ticks and stable:
            self.active=False;self.poses.clear();self.plateau.clear()
            return {'state':'released',**detail}
        if self.hold_ticks>=self.maximum_hold_ticks:
            self.failed=True
            return {'state':'failed',**detail}
        return {'state':'hold',**detail}

    def report(self):
        return {'episodes':self.episodes,'total_hold_ticks':self.total_hold_ticks,
                'failed':self.failed,'trigger_degrees':self.trigger_degrees,
                'release_degrees':self.release_degrees,
                'stable_plateau_max_degrees':15.,
                'minimum_hold_ticks':self.minimum_hold_ticks,
                'stable_window_ticks':self.poses.maxlen,
                'maximum_hold_ticks':self.maximum_hold_ticks}
