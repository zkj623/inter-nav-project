import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_stop_feedback import ForwardStopFeedback


def free_checker(target,limiter,now):return limiter.preview(target,now),True


def test_forward_hold_persists_across_zero_command_handoffs():
    servo=ForwardStopFeedback()
    servo.apply([.5,0,0],[.5,0,0],0.)
    first=servo.apply([.49,0,0],[.4,0,0],.01,stop_requested=True,checker=free_checker)
    assert servo.active and 0<first[0]<.5
    previous=first
    for i in range(2,160):
        command=servo.apply([0,0,0],[.2,.02,0],i*.01,checker=free_checker)
        assert servo.active
        assert np.linalg.norm(command[:2]-previous[:2])<=.65*.01+1e-9
        previous=command
    assert command[0]<0
    resumed=servo.apply([.2,0,0],[0,0,0],1.60,checker=free_checker)
    assert not servo.active and servo.rejoining
    assert np.linalg.norm(resumed[:2]-previous[:2])<=.65*.01+1e-9


def test_lateral_yaw_and_reverse_stops_are_not_armed():
    for request in [[0,.35,0],[0,0,1.],[-.35,0,0],[.2,.18,.8]]:
        servo=ForwardStopFeedback();servo.apply(request,request,0.)
        np.testing.assert_array_equal(servo.apply([0,0,0],request,.01,stop_requested=True),[0,0,0])
        assert not servo.active


def test_contact_mode_and_emergency_do_not_continue_feedback():
    servo=ForwardStopFeedback();servo.apply([.5,0,0],[.5,0,0],0.)
    servo.apply([0,0,0],[.4,0,0],.01,stop_requested=True,checker=free_checker)
    assert servo.active
    servo.apply([0,0,0],[.3,0,0],.02,feedback_allowed=False,checker=free_checker)
    assert not servo.active and not servo.eligible
    np.testing.assert_array_equal(servo.apply([.5,0,0],[.3,0,0],.03,healthy=False),[0,0,0])
    assert not servo.active and not servo.rejoining


def test_feedback_requires_and_respects_geometry_check():
    servo=ForwardStopFeedback();servo.apply([.5,0,0],[.5,0,0],0.)
    with pytest.raises(ValueError,match='checker'):
        servo.apply([0,0,0],[.3,0,0],.01,stop_requested=True)
    calls=[]
    def blocked(target,limiter,now):calls.append(target.copy());return np.ones(3),False
    np.testing.assert_array_equal(servo.apply([0,0,0],[.3,.2,.3],.02,checker=blocked),[0,0,0])
    assert len(calls)==1 and not servo.last_detail['motion_safe']
    assert np.linalg.norm(calls[0][:2])<=.4+1e-9 and abs(calls[0][2])<=.6


def test_small_forward_residue_does_not_arm_after_a_lateral_maneuver():
    servo=ForwardStopFeedback();servo.apply([0,.35,0],[0,.35,0],0.)
    servo.apply([.01,0,0],[0,.02,0],.1)
    np.testing.assert_array_equal(servo.apply([0,0,0],[0,.02,0],.11,stop_requested=True),[0,0,0])
    assert not servo.active
    servo.apply([.5,0,0],[.4,0,0],.2)
    servo.apply([.1,0,0],[.2,0,0],.3)
    servo.apply([0,0,0],[.2,0,0],.31,stop_requested=True,checker=free_checker)
    assert servo.active
