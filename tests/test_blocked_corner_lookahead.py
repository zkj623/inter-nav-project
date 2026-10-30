from pathlib import Path
import math
import numpy as np

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter


def observed_corner(stalled=True,empty=False):
    with np.load(Path(__file__).parent/'data/t04_blocked_corner_v140.npz') as data:
        free=data['free'].copy();limits=data['limits'].tolist()
        position=data['position'][:2].copy();q=data['orientation'].copy()
        path=data['path'].copy();index=int(data['waypoint_index'])
        start=data['planned_start'].copy();command=data['command'].copy()
    if empty:free[:]=True
    planner=SidePushPlanner(free,limits,resolution=.05)
    planner.navigation_route_radius=.30;planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.control_time=100.
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    limiter=VelocitySlewLimiter();limiter.command=command;limiter.last_time=99.992
    limiter.progress_position=position.copy();limiter.progress_time=98. if stalled else 100.
    tracker=SmoothExplorationTracker(start,path,limiter);tracker._waypoint_index=index
    return planner,position,heading,tracker


def test_blocked_corner_is_prepared_before_the_unreachable_waypoint_radius():
    planner,position,heading,tracker=observed_corner()
    corner=np.asarray(tracker._planned_path[tracker._waypoint_index])
    assert .28<np.linalg.norm(corner-position)<tracker.velocity_lookahead_distance
    before=planner.free.copy();path=tracker._planned_path
    _,state=tracker.action(position,heading,planner,[])
    assert state.get('blocked_corner_heading_rad')==0.
    assert state.get('local_staging') is True
    assert abs(state['target_heading_rad'])<1e-8
    np.testing.assert_array_equal(planner.free,before)
    assert tracker._planned_path==path


def test_normal_initial_acceleration_does_not_count_as_a_blocked_corner():
    planner,position,heading,tracker=observed_corner(stalled=False)
    _,state=tracker.action(position,heading,planner,[])
    assert 'blocked_corner_heading_rad' not in state
    assert not state.get('local_staging',False)


def test_clear_approach_keeps_normal_lookahead_tracking():
    planner,position,heading,tracker=observed_corner(empty=True)
    _,state=tracker.action(position,heading,planner,[])
    assert 'blocked_corner_heading_rad' not in state
    assert not state.get('local_staging',False)
