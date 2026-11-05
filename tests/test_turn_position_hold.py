import math
import numpy as np
from grutopia_extension.interactive_navigation.local_staging_control import TurnPositionHold


def test_hold_opposes_base_drift_without_changing_yaw_request():
    hold=TurnPositionHold()
    hold.apply([0,0,1],[0,0],0.,[0,0,0])
    request=hold.apply([0,0,1],[.03,-.02],0.,[.05,-.03,1.])
    assert request[0]<0 and request[1]>0 and request[2]==1.
    assert np.linalg.norm(request[:2])<=.15+1e-9


def test_hold_rotates_world_position_error_into_body_frame():
    hold=TurnPositionHold();hold.apply([0,0,1],[0,0],math.pi/2,[0,0,0])
    request=hold.apply([0,0,1],[.02,0],math.pi/2,[0,0,0])
    assert abs(request[0])<1e-8 and request[1]>0


def test_stop_and_intentional_arc_are_not_fought_by_position_hold():
    hold=TurnPositionHold();hold.apply([0,0,1],[0,0],0.,[0,0,0])
    np.testing.assert_array_equal(hold.apply([0,0,0],[.1,0],0.,[.1,0,0]),[0,0,0])
    assert hold.anchor is None
    np.testing.assert_array_equal(hold.apply([-.2,0,1],[.1,0],0.,[0,0,0]),[-.2,0,1])
    assert hold.anchor is None
