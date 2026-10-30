"""A geometry-safe command must not hide a physically unexecuted frontier."""
from pathlib import Path
import dataclasses
import json
import math

import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter
from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector

def frozen_stall(*,fresh=False,measured=(0.,0.,0.)):
    with np.load(Path(__file__).parent/'data/t05_execution_stall_v161.npz') as data:
        free=data['free'].copy();observed=data['observed'].copy();box_free=data['box_free'].copy()
        p=data['position'][:2].copy();q=data['orientation'].copy();q/=np.linalg.norm(q)
        path=data['path'].copy();start=data['planned_start'].copy();index=int(data['waypoint_index'])
        now=float(data['time_s']);command=data['command'].copy();limits=data['limits'].tolist()
        fields={field.name for field in dataclasses.fields(BoxState)}
        boxes=[BoxState(**{k:v for k,v in box.items() if k in fields}) for box in json.loads(str(data['boxes_json']))]
    planner=SidePushPlanner(free,limits,resolution=.05)
    planner.box_free=box_free;planner.planning_observed=observed;planner.planning_scope='local'
    planner.navigation_route_radius=.30;planner.navigation_half_width=.22
    planner.full_navigation_footprint=True;planner.layered_local_control=True
    limiter=VelocitySlewLimiter();limiter.command=command;limiter.last_time=now-.008
    limiter.progress_position=p.copy();limiter.progress_time=now if fresh else now-10.
    limiter.observe(measured,now)
    tracker=SmoothExplorationTracker(start,path,limiter);tracker._waypoint_index=index
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    def advance(tick):
        planner.control_time=now+tick*.008;limiter.observe(measured,planner.control_time)
        return tracker.action(p,heading,planner,boxes)
    return planner,tracker,advance

def test_recorded_ineffective_twist_reports_failure_and_rejects_only_its_frontier():
    planner,tracker,advance=frozen_stall();before=planner.free.copy();observed=planner.planning_observed.copy()
    for tick in range(40):
        action,state=advance(tick)
        assert np.linalg.norm(action['move_by_speed'][:2])<.16  # No speed increase to bypass collision checks.
    assert state.get('execution_stalled') is True
    assert state['needs_replan'] and state['blocked_ticks']>30
    selector=OnlineGoalSelector();endpoint=tracker._planned_path[-1]
    failure=selector.consume_execution_failure(endpoint,int(observed.sum()),state)
    assert failure['reason']=='no_measured_translation'
    assert selector.key(endpoint) in selector.visited
    assert selector.key([3.3,1.4]) not in selector.visited
    np.testing.assert_array_equal(planner.free,before)
    np.testing.assert_array_equal(planner.planning_observed,observed)

def test_startup_acceleration_is_not_execution_failure():
    _,_,advance=frozen_stall(fresh=True)
    for tick in range(40):
        _,state=advance(tick)
        assert not state.get('execution_stalled',False)
        assert not state['needs_replan']

def test_slow_measured_translation_is_not_classified_as_no_motion():
    _,_,advance=frozen_stall(measured=(.03,0.,0.))
    for tick in range(40):
        _,state=advance(tick)
        assert not state.get('execution_stalled',False)
        assert not state['needs_replan']

def test_measured_turning_is_not_classified_as_no_motion():
    _,_,advance=frozen_stall(measured=(0.,0.,.10))
    for tick in range(40):
        _,state=advance(tick)
        assert not state.get('execution_stalled',False)
