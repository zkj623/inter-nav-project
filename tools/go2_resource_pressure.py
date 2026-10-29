"""Pure resource-guard decisions for a single capped simulation process group."""
import math


class MemoryPressureGuard:
    def __init__(self, pause_below_gib=16., emergency_below_gib=4., resume_at_gib=56.,
                 low_seconds=10., resume_seconds=30., maximum_pause_seconds=600., abort_grace_seconds=300.):
        if not 0<emergency_below_gib<pause_below_gib<resume_at_gib:
            raise ValueError('Invalid memory thresholds')
        if min(low_seconds,resume_seconds,maximum_pause_seconds,abort_grace_seconds)<=0:
            raise ValueError('Resource durations must be positive')
        self.pause_below_gib=float(pause_below_gib);self.emergency_below_gib=float(emergency_below_gib)
        self.resume_at_gib=float(resume_at_gib);self.low_seconds=float(low_seconds)
        self.resume_seconds=float(resume_seconds);self.maximum_pause_seconds=float(maximum_pause_seconds)
        self.abort_grace_seconds=float(abort_grace_seconds)
        self.state='running';self.low_since=None;self.paused_since=None;self.recovered_since=None
        self.abort_since=None;self.last_time=None;self.pause_count=0;self.resume_count=0

    def update(self, now, available_gib):
        now=float(now);available_gib=float(available_gib)
        if not math.isfinite(now) or not math.isfinite(available_gib) or available_gib<0:
            raise ValueError('Finite time and nonnegative available memory required')
        if self.last_time is not None and now<self.last_time:
            raise ValueError('Monotonic resource time required')
        self.last_time=now
        if self.state=='killed':return None
        if available_gib<self.emergency_below_gib:
            self.state='killed';return {'action':'kill','reason':'host_memory_emergency'}
        if self.state=='aborting':
            if now-self.abort_since>=self.abort_grace_seconds:
                self.state='killed';return {'action':'kill','reason':'resource_abort_grace_expired'}
            return None
        if self.state=='paused':
            if available_gib>=self.resume_at_gib:
                if self.recovered_since is None:self.recovered_since=now
                if now-self.recovered_since>=self.resume_seconds:
                    self.state='running';self.low_since=None;self.paused_since=None;self.recovered_since=None
                    self.resume_count+=1;return {'action':'resume','reason':'host_memory_recovered'}
            else:
                self.recovered_since=None
                if now-self.paused_since>=self.maximum_pause_seconds:
                    self.state='aborting';self.abort_since=now
                    return {'action':'abort','reason':'resource_pause_timeout'}
            return None
        if available_gib<self.pause_below_gib:
            if self.low_since is None:self.low_since=now
            if now-self.low_since>=self.low_seconds:
                self.state='paused';self.paused_since=now;self.recovered_since=None;self.pause_count+=1
                return {'action':'pause','reason':'sustained_host_memory_pressure'}
        else:self.low_since=None
        return None

    def report(self):
        return {'state':self.state,'pause_count':self.pause_count,'resume_count':self.resume_count,
            'pause_below_gib':self.pause_below_gib,'emergency_below_gib':self.emergency_below_gib,
            'resume_at_gib':self.resume_at_gib,'low_seconds':self.low_seconds,
            'resume_seconds':self.resume_seconds,'maximum_pause_seconds':self.maximum_pause_seconds,
            'abort_grace_seconds':self.abort_grace_seconds}
