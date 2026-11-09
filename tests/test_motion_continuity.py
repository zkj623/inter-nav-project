"""Safety and behavior checks for the post-v197 candidate, not physics acceptance."""
import math
import numpy as np
import pytest

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_passage_guidance import PassageEntryGuide
from grutopia_extension.interactive_navigation.local_staging_control import TurnReversalGate
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker, VelocitySlewLimiter


def open_planner():
    planner=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    planner.navigation_half_width=.25
    return planner


def guide_fixture():
    planner=open_planner()
    old=BoxState('box',(0.,0.),(.8,.6,.6))
    moved=BoxState('box',(1.2,0.),(.8,.6,.6))
    guide=PassageEntryGuide({'corridor_start':[0,-1],'corridor_end':[0,1],
        'center':[0,0],'axis':[0,1],'side_parked':True,
        'original_box_would_block_corridor':True,'original_box':vars(old),
        'confirming_observation_time_s':10.,'last_box_motion_time_s':9.},[0,2])
    return planner,guide,[moved]


def join(planner,guide,boxes,**kwargs):
    values=dict(position=[0,-.1,.34],heading=0.,linear_velocity=[0,0,0],angular_velocity=[0,0,0],tilt=0.)
    values.update(kwargs)
    return guide.try_join_from_pose(planner=planner,boxes=boxes,**values)


def test_checked_current_pose_join_avoids_forced_one_metre_entry_backtrack():
    planner,guide,boxes=guide_fixture()
    detail=join(planner,guide,boxes)
    assert detail['joined'] and guide.entry_reached
    np.testing.assert_allclose(guide.target,[0,2])
    assert detail['connector_length_m']==pytest.approx(1.1)


def test_single_unknown_row_on_connector_cannot_be_skipped():
    planner,guide,boxes=guide_fixture()
    planner.free[planner.cell([0,.5])[0],:]=False
    before=planner.free.copy()
    assert not join(planner,guide,boxes)['joined']
    assert not guide.entry_reached
    np.testing.assert_array_equal(planner.free,before)


@pytest.mark.parametrize('change',[{'linear_velocity':[.07,0,0]}, {'angular_velocity':[0,0,.12]}, {'tilt':16.}])
def test_current_pose_join_requires_quiet_healthy_release(change):
    planner,guide,boxes=guide_fixture()
    assert not join(planner,guide,boxes,**change)['joined']
    assert not guide.entry_reached


def test_old_scan_cannot_admit_current_pose_join():
    planner,guide,boxes=guide_fixture()
    guide.certificate['confirming_observation_time_s']=9.
    assert not join(planner,guide,boxes)['joined']


def test_clear_connector_does_not_override_disconnected_goal():
    planner,guide,boxes=guide_fixture()
    planner.free[planner.cell([0,1.6])[0],:]=False
    assert not join(planner,guide,boxes)['joined']
    assert not guide.entry_reached


def test_nearer_entry_remains_on_approach_side_and_requires_arrival():
    planner,guide,boxes=guide_fixture()
    result=guide.shorten_checked_entry([0,-.1],0.,planner,boxes)
    assert result['shortened']
    assert guide.entry[1]<=-.56+1e-9
    assert guide.entry[1]>-1.
    assert not guide.entry_reached


def test_nearer_entry_does_not_cross_unknown_space():
    planner,guide,boxes=guide_fixture()
    planner.free[planner.cell([0,-.4])[0],:]=False
    before=guide.entry.copy()
    assert not guide.shorten_checked_entry([0,-.1],0.,planner,boxes)['shortened']
    np.testing.assert_array_equal(guide.entry,before)


def test_short_reverse_rate_oscillation_does_not_stop_a_large_turn():
    gate=TurnReversalGate()
    for cycle in range(10):
        assert not gate.update(1.,-.2,cycle*.1,cycle*.2)
        assert not gate.update(1.,-.2,cycle*.1-.01,cycle*.2+.05)
        assert not gate.update(1.,.4,cycle*.1+.03,cycle*.2+.10)


def test_sustained_actual_reverse_heading_still_brakes():
    gate=TurnReversalGate()
    assert not gate.update(1.,-.3,0.,0.)
    assert gate.update(1.,-.3,-.07,.25)


def test_large_reverse_excursion_brakes_without_waiting_for_timer():
    gate=TurnReversalGate()
    assert not gate.update(1.,-1.,0.,0.)
    assert gate.update(1.,-1.,-.09,.09)


def test_unitree_stalled_blended_motion_prepares_heading_before_switching_frontier():
    planner=open_planner();planner.control_time=10.
    limiter=VelocitySlewLimiter()
    limiter.command=np.array([.08,-.04,-.10]);limiter.policy_command=limiter.command.copy()
    limiter.last_time=9.99;limiter.progress_position=np.zeros(2);limiter.progress_time=0.
    limiter.observe([0,0,0],10.)
    tracker=SmoothExplorationTracker([0,0],[[2,-1]],limiter)
    before=limiter.command.copy()
    action,state=tracker.action(np.zeros(2),0.,planner,[])
    assert state.get('local_execution_recovery')=='measured_stall_prepare_heading'
    assert tracker.staging_active and not state['needs_replan']
    assert np.linalg.norm(np.asarray(action['move_by_speed'])[:2]-before[:2])<=.65*.01+1e-8


def test_old_policy_keeps_existing_stall_handling():
    planner=open_planner();planner.control_time=10.
    limiter=VelocitySlewLimiter();limiter.progress_position=np.zeros(2);limiter.progress_time=0.
    limiter.observe([0,0,0],10.)
    tracker=SmoothExplorationTracker([0,0],[[2,-1]],limiter)
    _,state=tracker.action(np.zeros(2),0.,planner,[])
    assert 'local_execution_recovery' not in state
