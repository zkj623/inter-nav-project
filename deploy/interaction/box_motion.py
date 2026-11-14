"""Source-timed settling evidence for an observed rigid box."""
from collections import deque

import numpy as np


class ObservedBoxMotion:
    """Bound measured box motion over independent scans, including rotation.

    SDK thresholds come from its explicit calibration profile. A stable result
    means motion is below those measured resolution/speed bounds, not exactly
    zero physical velocity. Repeated source frames cannot extend the window.
    """
    def __init__(self,profile):
        self.profile=profile;self.samples=deque();self.seen_ids=set()
        self.anchor=None;self.last_motion_at=None;self.last_motion_id=None
        self.latest_at=None;self.latest_id=None;self.latest_corners=None
        self.settled=False;self.center_velocity=np.zeros(2);self.diagnostic={}

    @staticmethod
    def _corners(box):
        dimensions=np.asarray(getattr(box,'local_dimensions',None),float)
        q=np.asarray(getattr(box,'orientation',None),float)
        z=getattr(box,'center_z',None)
        xy=np.asarray(box.xy,float)
        if (dimensions.shape!=(3,) or q.shape!=(4,) or xy.shape!=(2,) or z is None
                or not np.isfinite([*dimensions,*q,*xy,z]).all()
                or np.any(dimensions<=0) or np.linalg.norm(q)<1e-9):
            raise ValueError('Box stability requires registered dimensions and measured rigid pose')
        w,x,y,zq=q/np.linalg.norm(q)
        rotation=np.array([[1-2*(y*y+zq*zq),2*(x*y-w*zq),2*(x*zq+w*y)],
            [2*(x*y+w*zq),1-2*(x*x+zq*zq),2*(y*zq-w*x)],
            [2*(x*zq-w*y),2*(y*zq+w*x),1-2*(x*x+y*y)]])
        corners=np.array([[a,b,c] for a in [-1.,1.] for b in [-1.,1.] for c in [-1.,1.]])
        return (corners*(dimensions/2))@rotation.T+np.array([*xy,float(z)])

    def observe(self,box,stamp,source_id,now):
        if (stamp is None or not isinstance(source_id,(str,int,np.integer)) or not str(source_id)
                or not np.isfinite(stamp) or not np.isfinite(now) or float(stamp)>float(now)+1e-6):
            raise ValueError('Box stability requires an original source timestamp and scan identity')
        stamp=float(stamp);source_id=str(source_id);corners=self._corners(box)
        if source_id==self.latest_id:
            if stamp!=self.latest_at or not np.allclose(corners,self.latest_corners,rtol=0.,atol=1e-9):
                raise ValueError('A repeated box source cannot change its measurement or timestamp')
            return self.diagnostic
        if source_id in self.seen_ids or (self.latest_at is not None and stamp<=self.latest_at):
            raise ValueError('Box stability source scans must advance monotonically')
        p=self.profile
        if self.latest_at is not None and stamp-self.latest_at>p.box_settle_seconds:
            self.samples.clear();self.anchor=None
        if self.anchor is None or np.max(np.linalg.norm(corners-self.anchor,axis=1))>p.box_position_tolerance_m:
            self.anchor=corners.copy();self.last_motion_at=stamp;self.last_motion_id=source_id
        self.samples.append((stamp,corners.copy()))
        self.seen_ids.add(source_id)
        self.latest_at=stamp;self.latest_id=source_id;self.latest_corners=corners.copy()
        cutoff=stamp-p.box_settle_seconds
        # Keep one sample at/before the boundary so a complete measured window
        # is required; never synthesize stationary samples across missing time.
        while len(self.samples)>2 and self.samples[1][0]<=cutoff+1e-9:self.samples.popleft()
        times=np.asarray([item[0] for item in self.samples],float)
        points=np.asarray([item[1] for item in self.samples],float)
        span=float(times[-1]-times[0]);centered=times-times.mean();denominator=float(centered@centered)
        velocity=(np.tensordot(centered,points,axes=(0,0))/denominator if denominator>1e-12 else np.zeros((8,3)))
        speed=float(np.max(np.linalg.norm(velocity,axis=1)))
        excursion=float(np.max(np.linalg.norm(points-points[0],axis=2)))
        self.center_velocity=velocity.mean(axis=0)[:2]
        motion=(speed>p.box_max_settled_speed_mps or excursion>p.box_position_tolerance_m)
        # A short window has not averaged the calibrated stationary jitter yet.
        # Accumulated anchor displacement still detects real movement immediately.
        full=len(self.samples)>=3 and span+1e-9>=p.box_settle_seconds
        if full and motion:self.last_motion_at=stamp;self.last_motion_id=source_id
        self.settled=bool(full and not motion)
        self.diagnostic={'settled':self.settled,'source_observed_at':stamp,'source_id':source_id,
            'last_motion_source_at':self.last_motion_at,'last_motion_source_id':self.last_motion_id,
            'window_seconds':span,'independent_scans':len(self.samples),
            'max_surface_displacement_m':excursion,'max_surface_speed_mps':speed,
            'position_tolerance_m':p.box_position_tolerance_m,'speed_tolerance_mps':p.box_max_settled_speed_mps,
            'required_window_seconds':p.box_settle_seconds,'calibration_reference':p.calibration_reference,
            'geometry':'full_registered_cuboid_at_measured_pose'}
        return self.diagnostic
