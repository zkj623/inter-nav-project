import numpy as np
from grutopia_extension.interactive_navigation.go2_interaction_motion import (
    navigation_recovery_enabled,MeasuredStopGate,ObservedSubgoalArrival)


def test_inspection_and_skill_subnavigation_have_attitude_protection():
    for phase in ['inspect_navigate','inspect_turn','inspect_wait','navigate','decide']:
        assert navigation_recovery_enabled(phase)
    assert navigation_recovery_enabled('skill','navigate_to_stage')
    assert navigation_recovery_enabled('skill','reobserve')
    assert not navigation_recovery_enabled('skill','push')
    assert not navigation_recovery_enabled('skill','engage')


def test_close_but_still_moving_or_rolling_does_not_handoff():
    gate=ObservedSubgoalArrival();p=[.1,0,.4]
    for time in [0.,.5,1.]:
        r=gate.update([0,0],p,[.11,0,0],[0,0,0],10.,time)
        assert r['subgoal_braking'] and not r['subgoal_reached']
    for time in [1.5,2.,2.5]:
        assert not gate.update([0,0],p,[0,0,0],[.24,0,0],12.,time)['subgoal_reached']
    assert not gate.update([0,0],p,[0,0,0],[0,0,0],10.,3.)['subgoal_reached']
    assert gate.update([0,0],p,[0,0,0],[0,0,0],10.,3.5)['subgoal_reached']


def test_braking_is_latched_through_drift_then_requests_a_fresh_route():
    gate=ObservedSubgoalArrival()
    gate.update([0,0],[.1,0,.4],[.15,0,0],[0,0,0],0.,0.)
    r=gate.update([0,0],[.2,0,.4],[.1,0,0],[0,0,0],0.,.5)
    assert r['subgoal_braking'] and not r['replan']
    gate.update([0,0],[.2,0,.4],[0,0,0],[0,0,0],0.,1.)
    r=gate.update([0,0],[.2,0,.4],[0,0,0],[0,0,0],0.,1.5)
    assert r['replan'] and not r['subgoal_reached']


def test_pose_drift_and_sensor_motion_restart_the_settling_window():
    gate=MeasuredStopGate()
    assert not gate.update([0,0,.4],np.zeros(3),np.zeros(3),5.,0.)
    assert not gate.update([.03,0,.4],np.zeros(3),np.zeros(3),5.,.5)
    assert not gate.update([.03,0,.38],np.zeros(3),np.zeros(3),5.,1.)
    assert not gate.update([.03,0,.38],np.zeros(3),np.zeros(3),8.,1.5)
    assert gate.update([.03,0,.38],np.zeros(3),np.zeros(3),8.,2.)


def test_skill_stop_is_slew_limited_but_failure_and_obstacle_stops_are_not_bypassed():
    from types import SimpleNamespace as NS
    from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_brake
    from grutopia_extension.interactive_navigation.interaction_skill import SkillCommand
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    limiter=VelocitySlewLimiter();limiter.accept([.25,0,0],0.);limiter.measured=np.array([.15,0,0])
    ctx=NS(position=np.array([0.,0.,.4]),heading=0.,planner=p,boxes=[],limiter=limiter,now=.008,healthy=True)
    command=SkillCommand('settle_contact')
    value=checked_skill_brake(command,ctx)
    assert 0.<value[0]<.25 and value[1]==value[2]==0.
    command.status='failed';command.velocity=(.2,0.,0.)
    np.testing.assert_array_equal(checked_skill_brake(command,ctx),[0,0,0])
    command.velocity=(0.,0.,0.)
    command.status='running';p.free[:,p.cell([.6,0])[1]:]=False
    np.testing.assert_array_equal(checked_skill_brake(command,ctx),[0,0,0])


def test_navigation_goal_is_handled_by_its_own_controller():
    from types import SimpleNamespace as NS
    from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_brake
    from grutopia_extension.interactive_navigation.interaction_skill import SkillCommand
    c=SkillCommand('navigate_to_stage',navigation_goal=(1.,2.))
    np.testing.assert_array_equal(checked_skill_brake(c,NS(limiter=object(),healthy=True)),[0,0,0])


def test_short_subgoal_keeps_an_effective_forward_request_until_arrival_gate():
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,SmoothExplorationTracker
    p=SidePushPlanner(np.ones((100,100),bool),[[-2.5,2.5],[-2.5,2.5]],resolution=.05)
    p.control_time=.1
    limiter=VelocitySlewLimiter();limiter.accept([.3,0,0],0.)
    tracker=SmoothExplorationTracker([0.,0.],[(.21,0.,0.)],limiter,final_goal_slow_radius=.30)
    action,state=tracker.action([0.,0.],0.,p,[])
    assert state['desired_command'][0]>=.25
    assert state['body_motion_safe'] and action['move_by_speed'][0]>0.


def test_subgoal_brakes_for_measured_forward_travel_but_still_requires_final_tolerance():
    gate=ObservedSubgoalArrival()
    moving=gate.update([.30,0.],[0.,0.,.4],[.4,0.,0.],[0.,0.,0.],0.,0.,heading=0.,command=[.5,0.,0.])
    assert moving['subgoal_braking'] and not moving['subgoal_reached']
    # The robot stopped too early: prediction is not permission to accept it.
    gate.update([.30,0.],[.10,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.,heading=0.,command=[0.,0.,0.])
    settled=gate.update([.30,0.],[.10,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.5,heading=0.,command=[0.,0.,0.])
    assert settled['replan'] and not settled['subgoal_reached']
    gate=ObservedSubgoalArrival()
    away=gate.update([.30,0.],[0.,0.,.4],[-.4,0.,0.],[0.,0.,0.],0.,0.,heading=0.,command=[-.5,0.,0.])
    assert not away['subgoal_braking']
