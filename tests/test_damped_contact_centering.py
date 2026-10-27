import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_contact_control import damped_contact_centering_velocity,contact_braking_required


@pytest.mark.parametrize('axis,delta,expected',[
    ('lateral',[.8,.4],[0,.25]),
    ('longitudinal',[.8,.4],[.25,0]),
    ('longitudinal',[-.8,.4],[-.25,0]),
])
def test_configured_speed_limits_both_axes_and_reverse(axis,delta,expected):
    np.testing.assert_allclose(damped_contact_centering_velocity(delta,[1,0],[0,0],.25,axis),expected)


def test_measured_motion_reduces_request_before_predictive_braking():
    slow=damped_contact_centering_velocity([0,.20],[1,0],[0,0],.30,'lateral')
    moving=damped_contact_centering_velocity([0,.20],[1,0],[0,.2],.30,'lateral')
    assert moving[1]<slow[1]
    assert moving[1]>=.20
    assert contact_braking_required([0,.06],[0,.2],horizon=.4)


def test_rotated_axis_keeps_orthogonal_error_out_of_the_command():
    np.testing.assert_allclose(damped_contact_centering_velocity([-.08,.4],[0,1],[0,0],.25,'lateral'),[-.20,0])
    np.testing.assert_array_equal(damped_contact_centering_velocity([0,.4],[0,1],[0,0],.25,'lateral'),[0,0])
