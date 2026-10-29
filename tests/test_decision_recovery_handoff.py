"""Regression for cancellation on the first momentarily free footprint."""
from pathlib import Path
import json
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ForwardExplorationPathTracker,decision_recovery_required,navigation_body_blocked,turn_reduces_overlap)
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,smooth_recovery_action


def setup_case():
    with np.load(Path(__file__).parent/'fixtures/t04_recovery_handoff_v96.npz') as z:
        planner=SidePushPlanner(z['free'].copy(),z['limits'].tolist(),resolution=.05)
        planner.planning_observed=z['observed'].copy()
        blocked=z['blocked_position'].copy();heading=float(z['blocked_heading'])
        free=z['free_position'].copy();free_heading=float(z['free_heading'])
        boxes=[BoxState(**r) for r in json.loads(str(z['boxes_json']))]
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.layered_local_control=True;planner.planning_scope='local';planner.navigation_route_radius=.30
    planner.control_time=0.
    tracker=ForwardExplorationPathTracker(blocked[:2],[(10.3,1.75)],guard_motion=False)
    assert decision_recovery_required(planner,boxes,blocked,heading)
    action,state=tracker.action(blocked[:2],heading,planner,boxes)
    assert state['recovering_buffer'] and not state['needs_replan']
    assert tracker._buffer_turn_target is not None
    return planner,boxes,tracker,free,free_heading


def test_briefly_free_recorded_pose_keeps_checked_turn_and_velocity_ramp():
    planner,boxes,tracker,position,heading=setup_case()
    target=tracker._buffer_turn_target
    # The old caller would discard this exact tracker at this free footprint.
    assert not navigation_body_blocked(planner,boxes,heading)[planner.cell(position)]
    assert turn_reduces_overlap(planner,boxes,position,heading,target)
    assert decision_recovery_required(planner,boxes,position,heading,tracker)
    limiter=VelocitySlewLimiter();limiter.accept([0.,0.,0.],0.)
    before=planner.free.copy();sent=[]
    for tick in range(1,81):
        planner.control_time=tick*.01
        assert decision_recovery_required(planner,boxes,position,heading,tracker)
        action,state=tracker.action(position[:2],heading,planner,boxes)
        command=smooth_recovery_action(action,position[:2],heading,planner,boxes,limiter,planner.control_time)
        limiter.accept(command['move_by_speed'],planner.control_time)
        sent.append(command['move_by_speed'][2])
        assert tracker._buffer_turn_target==target
    assert abs(sent[-1])>=.79
    assert max(abs(np.diff(sent)))<=.018+1e-8
    np.testing.assert_array_equal(planner.free,before)


def test_new_obstacle_still_stops_pending_turn_then_requests_replan():
    planner,boxes,tracker,position,heading=setup_case()
    planner.free[:]=False
    planner.control_time=.1
    action,state=tracker.action(position[:2],heading,planner,boxes)
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
    assert state['controller']=='recovery_turn_observation_hold'
    planner.control_time=1.2
    action,state=tracker.action(position[:2],heading,planner,boxes)
    assert state['needs_replan'] and tracker._buffer_turn_target is None
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])


def test_reaching_checked_heading_releases_pending_turn():
    planner,boxes,tracker,position,_=setup_case()
    target=tracker._buffer_turn_target
    planner.control_time=.1
    action,state=tracker.action(position[:2],target,planner,boxes)
    assert state['needs_replan'] and tracker._buffer_turn_target is None
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
    assert not decision_recovery_required(planner,boxes,position,target,tracker)
