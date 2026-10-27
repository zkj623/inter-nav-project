import math
import numpy as np

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl, checked_contact_turn_command
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear, body_rotation_clear


def planner():
    p = SidePushPlanner(np.ones((120, 120), bool), [[-3, 3], [-3, 3]], resolution=.05)
    p.navigation_half_width=.22
    return p


def test_stationary_yaw_is_rejected_when_measured_drift_reaches_wall():
    p=planner(); p.free[72:,:]=False  # wall y=0.6, outside stationary turning footprint
    limiter=VelocitySlewLimiter(); limiter.accept([0,0,1],0)
    limiter.measured=np.array([0,.5,0])
    assert body_rotation_clear(p,[],np.zeros(2),0,math.pi)
    command,safe=checked_contact_turn_command([0,0,1],np.zeros(2),0,p,[],limiter,.01)
    assert not safe
    np.testing.assert_array_equal(command,[0,0,0])


def test_contact_turn_moves_to_space_before_rotating_near_wall():
    p=planner(); p.free[68:,:]=False
    controller=ContactTurnControl(np.zeros(2),0,math.pi,p,[],0)
    assert not controller.failed
    assert controller.stage.target[1]<0
    assert body_translation_recovery_clear(p,[],np.zeros(2),0,controller.stage.target)
    assert body_rotation_clear(p,[],controller.stage.target,0,controller.stage.goal_heading)
    limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0)
    command,state=controller.update(np.zeros(2),0,p,[],limiter,.01)
    assert state['staging_phase']=='staging_move'
    assert command[1]<0 and command[2]==0


def test_turn_request_is_slew_limited_and_unknown_remains_blocked():
    p=planner();limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0)
    command,safe=checked_contact_turn_command([0,0,1],np.zeros(2),0,p,[],limiter,.01)
    assert safe and abs(command[2]-.018)<1e-10
    p.free[60:,:]=False
    before=p.free.copy()
    command,safe=checked_contact_turn_command([0,0,1],np.zeros(2),0,p,[],limiter,.01)
    assert not safe
    np.testing.assert_array_equal(command,[0,0,0])
    np.testing.assert_array_equal(p.free,before)
