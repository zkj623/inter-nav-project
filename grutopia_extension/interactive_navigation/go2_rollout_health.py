"""Audit every environment tick, including centering and goal-settling pauses."""
import math
import numpy as np


class Go2RolloutHealth:
    def __init__(self,floor_height,maximum_tilt=30.,minimum_height=.20):
        self.floor_height=float(floor_height);self.tilt_limit=float(maximum_tilt)
        self.height_limit=self.floor_height+minimum_height
        self.max_tilt=0.;self.min_height=float('inf');self.samples=0;self.violation=None

    @property
    def healthy(self):return self.violation is None

    def observe(self,position,orientation):
        self.samples+=1
        p=np.asarray(position,float);q=np.asarray(orientation,float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
            if self.violation is None:self.violation={'reason':'nonfinite_pose','sample':self.samples}
            return
        q=q/np.linalg.norm(q)
        tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
        self.max_tilt=max(self.max_tilt,tilt);self.min_height=min(self.min_height,float(p[2]))
        if self.violation is None and (tilt>self.tilt_limit or p[2]<self.height_limit):
            self.violation={'reason':'tilt_limit' if tilt>self.tilt_limit else 'base_too_low',
                'sample':self.samples,'tilt_degrees':tilt,'position':p.tolist()}

    def report(self):
        return {'samples':self.samples,'healthy':self.healthy,'maximum_tilt_degrees':self.max_tilt,
            'minimum_base_height_m':self.min_height if math.isfinite(self.min_height) else None,
            'first_violation':self.violation,'scope':'all rollout ticks including settling'}
