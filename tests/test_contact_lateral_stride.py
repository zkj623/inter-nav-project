import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_contact_control import damped_contact_centering_velocity


def test_lateral_stride_does_not_decay_into_measured_no_step_region():
    np.testing.assert_allclose(damped_contact_centering_velocity([.2,.06],[1,0],[0,.2],.25,'lateral',lateral_speed=.35),[0,.35])
    np.testing.assert_allclose(damped_contact_centering_velocity([.2,-.06],[1,0],[0,-.2],.25,'lateral',lateral_speed=.35),[0,-.35])
    forward=damped_contact_centering_velocity([.5,.3],[1,0],[0,0],.25,'longitudinal',lateral_speed=.35)
    np.testing.assert_allclose(forward,[.25,0])


def test_subthreshold_lateral_configuration_is_rejected():
    with pytest.raises(ValueError,match='lateral'):
        damped_contact_centering_velocity([0,.2],[1,0],[0,0],.25,'lateral',lateral_speed=.25)
