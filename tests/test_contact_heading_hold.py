import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import contact_heading_hold,ContactYawServo


def test_small_centering_drift_is_corrected_and_yaw_request_is_bounded():
    assert contact_heading_hold(0,0)==0
    assert contact_heading_hold(-.08,.1)<-.24
    assert contact_heading_hold(.2,-.3)==.35
    assert contact_heading_hold(-.2,.3)==-.35


def test_heading_feedback_rejects_constant_sidestep_bias_in_simple_response_model():
    # A diagnostic control-model check, not a substitute for the Go2 rollout.
    heading=rate=0.;dt=.005
    for _ in range(800):
        request=contact_heading_hold(-heading,rate)
        rate+=dt/.15*(.5*request+.1-rate)
        heading+=dt*rate
    assert abs(heading)<.08
    assert abs(rate)<.01


def test_fine_yaw_brakes_on_measured_motion_without_reversing_the_moving_gait():
    control=ContactYawServo(.1)
    assert 0<control.update(.15,0,0)<=.65
    assert control.update(.08,.20,.1)==0
    assert control.update(-.08,.15,.2)==0
    assert control.update(-.12,0,.3)==0
    assert control.update(-.12,0,.55)==0
    assert control.update(-.12,0,.56)<0


def test_fine_yaw_stops_inside_contact_tolerance_in_delayed_response_model():
    control=ContactYawServo(.1);heading=.16;rate=0.;dt=.005
    for i in range(1200):
        command=control.update(-heading,rate,i*dt)
        rate+=dt/.20*(.5*command-rate)
        heading+=rate*dt
    assert abs(heading)<.1 and abs(rate)<.01
