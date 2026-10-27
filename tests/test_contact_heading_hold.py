import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import contact_heading_hold


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
