"""Exercise the tracker/intent/servo chain; these are not dynamics tests."""
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_stop_feedback import ForwardStopFeedback,with_staging_stop_request,tracked_navigation_action
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,SmoothExplorationTracker
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command


def staged_action(enabled=False):
    planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True;planner.control_time=.01
    limiter=VelocitySlewLimiter();limiter.accept([.4,0,0],0.)
    limiter.observe([.4,0,0],0.)
    tracker=SmoothExplorationTracker([0,0],[[-1,1]],limiter)
    action,state=tracked_navigation_action(tracker,np.zeros(2),0.,planner,[],enabled=enabled)
    assert state['staging_phase']=='staging_brake'
    return planner,action,state


def test_pre_turn_forward_brake_arms_checked_feedback_before_request_reaches_zero():
    planner,action,state=staged_action(enabled=True)
    assert 0.<action['move_by_speed'][0]<.4
    servo=ForwardStopFeedback();servo.apply([.4,0,0],[.4,0,0],0.)
    calls=[]
    def checked(target,limiter,now):
        calls.append(target.copy())
        return checked_contact_turn_command(target,np.zeros(3),0.,planner,[],limiter,now)
    sent=servo.apply(action['move_by_speed'],[.4,0,0],.01,
                     stop_requested=action.get('_forward_stop_request',False),checker=checked)
    assert servo.active and calls and calls[0][0]<0.
    assert 0.<sent[0]<.4
    previous=sent
    # A stop remains active across zero-command state handoffs. New motion
    # releases it through the existing acceleration-checked rejoin.
    for tick in range(2,101):
        sent=servo.apply([0,0,0],[.12,0,0],tick*.01,checker=checked)
        assert servo.active
        assert np.linalg.norm(sent[:2]-previous[:2])<=.65*.01+1e-8
        previous=sent
    servo.apply([0,0,1.],[0,0,0],1.01,checker=checked)
    assert not servo.active


@pytest.mark.parametrize('prior',[[0,.35,0],[0,0,1.],[-.35,0,0]])
def test_staging_stop_does_not_expand_servo_travel_eligibility(prior):
    _,_,state=staged_action()
    action=with_staging_stop_request({'move_by_speed':[0,0,0]},state,enabled=True)
    servo=ForwardStopFeedback();servo.apply(prior,prior,0.)
    sent=servo.apply(action['move_by_speed'],prior,.01,
                     stop_requested=action['_forward_stop_request'])
    assert not servo.active
    np.testing.assert_array_equal(sent,[0,0,0])


def test_disabled_feedback_and_collision_stops_do_not_gain_deliberate_stop_intent():
    _,raw,state=staged_action()
    assert with_staging_stop_request(raw,state,enabled=False) is raw
    variants=[{'body_motion_safe':False},{'needs_replan':True},
              {'emergency_collision_stop':True},{'measured_braking_space_blocked':True},
              {'staging_phase':'staging_turn_brake'},{'staging_phase':'staging_move'},
              {'staging_phase':'staging_turn'},{'local_staging':False}]
    for changed in variants:
        assert with_staging_stop_request(raw,{**state,**changed},enabled=True) is raw


def test_staging_feedback_keeps_current_map_veto():
    planner,raw,state=staged_action()
    action=with_staging_stop_request(raw,state,enabled=True)
    servo=ForwardStopFeedback();servo.apply([.4,0,0],[.4,0,0],0.)
    planner.free[:]=False  # A new local scan invalidates the prior safe request.
    def checked(target,limiter,now):
        return checked_contact_turn_command(target,np.zeros(3),0.,planner,[],limiter,now)
    sent=servo.apply(action['move_by_speed'],[.4,0,0],.01,
                     stop_requested=action['_forward_stop_request'],checker=checked)
    np.testing.assert_array_equal(sent,[0,0,0])
    assert not servo.last_detail['motion_safe']
