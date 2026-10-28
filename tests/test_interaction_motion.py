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
