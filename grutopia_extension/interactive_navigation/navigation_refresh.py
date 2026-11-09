"""Reconsider exploration after poor route progress, including repeated staging."""
import numpy as np
from grutopia_extension.interactive_navigation.route_progress import RouteProgress


def passage_reference_target(connection,guide,goal):
    """Use the current checked operation, including guide-based fallbacks."""
    if connection is not None:
        for operation in connection.plan['operations'][connection.index:]:
            if operation['kind']=='move':return operation['target']
        return goal
    if guide is not None:return guide.target
    return None

def rolling_refresh_allowed(reason,staging,measured):
    return (reason in ['task_route_became_available','better_observed_frontier'] and not staging
            and measured[0]>.08 and abs(measured[2])<.30)

def reconsider_frontier(pending,selector,layers,position,goal,endpoint,now,staging,translation_grace=False):
    """Finish one bounded maneuver before opportunistic goal changes.

    Reference display remains current. A no-progress watchdog request still
    interrupts repeated ineffective staging, preserving its existing budgets.
    """
    if pending is not None:
        selector.defer_frontier(endpoint,now)
        return pending
    if staging or translation_grace:
        selector.refresh_reference(layers,position,endpoint,now)
        return None
    return selector.reconsider(layers,position,goal,endpoint,now)

class NavigationProgressWatch:
    def __init__(self):
        self.key=None;self.route=None;self.index=0;self.checkpoint=0.;self.position=None
        self.last_progress=0.;self.stage=None;self.stage_count=0
        self.completion_credit_used=False;self.translation_until=None

    def set_route(self,start,path,endpoint,now):
        key=tuple(np.rint(np.asarray(endpoint[:2])/.4).astype(int))
        if self.key!=key or self.position is None:
            self.last_progress=float(now);self.position=np.asarray(start[:2],float).copy();self.stage_count=0
            self.completion_credit_used=False;self.translation_until=None
        self.key=key;self.route=RouteProgress(start,path);self.index=0;self.checkpoint=0.;self.stage=None

    def observe_stage(self,stage,now=None):
        if (stage is None and self.stage is not None and not getattr(self.stage,'failed',False)
                and now is not None and self.stage_count==1 and not self.completion_credit_used):
            # One bounded initial turn must get a chance to produce translation.
            # Do not reset the actual progress clock or grant repeated credits.
            self.translation_until=float(now)+2.;self.completion_credit_used=True
        if stage is not None and stage is not self.stage:self.stage_count+=1
        self.stage=stage

    def translation_grace(self,now):
        return self.stage is None and self.translation_until is not None and now<self.translation_until

    def check(self,position,now,clear,staging=False):
        if self.route is None:return None
        position=np.asarray(position[:2],float)
        self.index=self.route.advance(position,self.index,clear)
        if self.route.progress-self.checkpoint>=.20 and np.linalg.norm(position-self.position)>=.15:
            self.checkpoint=self.route.progress;self.position=position.copy();self.last_progress=float(now);self.stage_count=0
            self.completion_credit_used=False;self.translation_until=None
        elapsed=float(now-self.last_progress)
        if not staging and self.translation_grace(now):return None
        if (not staging and elapsed>=6.) or (staging and self.stage_count>=2 and elapsed>=8.):
            return {'reason':'no_forward_route_progress','without_progress_s':elapsed,
                    'forward_route_progress_m':self.route.progress-self.checkpoint,
                    'staging_episodes_since_progress':self.stage_count}
        return None
