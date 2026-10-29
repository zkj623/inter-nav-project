"""Recovery must not restart while the body is bouncing or still rotating."""
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_recovery_settling import RecoveryMotionGate
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,smooth_recovery_action
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def state(now,linear=(0,0,0),angular=(0,0,0),position=(0,0,.42),tilt=0.):
    return (np.array(position,float),np.array(linear,float),np.array(angular,float),float(tilt),float(now))


def test_recorded_bounce_cannot_cancel_an_exit_hold_when_map_becomes_blocked_again():
    gate=RecoveryMotionGate();request=[.2,0,0]
    np.testing.assert_array_equal(gate.filter(request,[0,0,0],40.,body_state=state(40.)),request)
    assert not gate.leave_recovery([.011,.017,.08],40.41,
        body_state=state(40.41,linear=(.011,.017,-.105),angular=(.357,.134,.08),tilt=12.07),command=[0,0,0])
    # Actual T04 v107 motion near40.81s had little planar movement but upward
    # speed .077m/s and pitch rate -.335rad/s. Same-mode recovery must wait.
    command=gate.filter(request,[-.031,.015,.069],40.81,
        body_state=state(40.81,linear=(-.031,.015,.077),angular=(-.107,-.335,.069),tilt=7.64),command=[0,0,0])
    np.testing.assert_array_equal(command,[0,0,0])
    np.testing.assert_array_equal(gate.filter(request,[0,0,0],41.,body_state=state(41.)),[0,0,0])
    np.testing.assert_array_equal(gate.filter(request,[0,0,0],41.21,body_state=state(41.21)),request)


@pytest.mark.parametrize('motion',[state(0.,linear=(0,0,.051)),state(0.,angular=(.101,0,0)),state(0.,angular=(0,.101,0)),state(0.,tilt=15.)])
def test_startup_does_not_treat_planar_quiet_as_complete_body_quiet(motion):
    gate=RecoveryMotionGate()
    np.testing.assert_array_equal(gate.filter([.2,0,0],[0,0,0],0.,body_state=motion),[0,0,0])


def test_zero_request_latches_a_stop_before_restarting_the_same_direction():
    gate=RecoveryMotionGate()
    gate.filter([.2,0,0],[0,0,0],0.,body_state=state(0.))
    gate.filter([0,0,0],[.1,0,0],.1,body_state=state(.1,linear=(.1,0,0)))
    np.testing.assert_array_equal(gate.filter([.2,0,0],[0,0,0],.2,body_state=state(.2)),[0,0,0])
    np.testing.assert_array_equal(gate.filter([.2,0,0],[0,0,0],.41,body_state=state(.41)),[.2,0,0])


def test_map_classification_changes_do_not_reset_a_valid_stopping_window():
    gate=RecoveryMotionGate()
    gate.filter([.2,0,0],[0,0,0],0.,body_state=state(0.))
    assert not gate.leave_recovery([0,0,0],1.,body_state=state(1.))
    np.testing.assert_array_equal(gate.filter([.2,0,0],[0,0,0],1.1,body_state=state(1.1)),[0,0,0])
    assert gate.leave_recovery([0,0,0],1.21,body_state=state(1.21))
    # Already measured-stop handoff does not introduce another idle dwell.
    assert gate.leave_recovery([0,0,0],1.22,body_state=state(1.22))


@pytest.mark.parametrize('changed',[state(.2,position=(.026,0,.42)),state(.2,position=(0,0,.436)),state(.2,tilt=1.6)])
def test_position_height_or_tilt_drift_restarts_the_stable_window(changed):
    gate=RecoveryMotionGate()
    assert not gate.leave_recovery([0,0,0],0.,body_state=state(0.))
    assert not gate.leave_recovery([0,0,0],.2,body_state=changed)
    stable=(*changed[:4],.41)
    assert gate.leave_recovery([0,0,0],.41,body_state=stable)


@pytest.mark.parametrize('command',[[.031,0,0],[0,0,.061]])
def test_commanded_motion_must_finish_even_if_measured_velocity_is_momentarily_zero(command):
    gate=RecoveryMotionGate()
    assert not gate.leave_recovery([0,0,0],0.,body_state=state(0.),command=command)
    assert not gate.leave_recovery([0,0,0],.3,body_state=state(.3),command=command)


def test_stale_or_nonfinite_full_feedback_cannot_complete_a_stop():
    gate=RecoveryMotionGate()
    assert not gate.leave_recovery([0,0,0],.5,body_state=state(0.))
    assert not gate.leave_recovery([0,0,0],.6,body_state=state(.6,linear=(0,0,float('nan'))))


def test_real_smoothing_entry_uses_full_feedback_and_guidance_reset_preserves_it():
    limiter=VelocitySlewLimiter()
    limiter.observe([0,0,0],0.,body_state=((0,0,.42),(0,0,.08),(0,0,0),0.))
    limiter.clear_guidance([0,0,.42],0.)
    planner=SidePushPlanner(np.ones((100,100),bool),[[-2.5,2.5],[-2.5,2.5]],resolution=.05)
    action=smooth_recovery_action({'move_by_speed':[.25,0,0]},[0,0,.42],0.,planner,[],limiter,.01)
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
