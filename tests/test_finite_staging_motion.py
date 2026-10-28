import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command


def test_finite_side_step_checks_goal_and_actual_braking_not_motion_past_goal():
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    p.navigation_half_width=.22
    p.free[p.cell([0,.45])[0]:,:]=False
    limiter=VelocitySlewLimiter();limiter.accept([0,.35,0],0)
    _,unbounded=checked_contact_turn_command([0,.35,0],[0,0],0,p,[],limiter,.01)
    assert not unbounded
    command,safe=checked_contact_turn_command([0,.35,0],[0,0],0,p,[],limiter,.01,target_position=[0,.15])
    assert safe
    np.testing.assert_allclose(command,[0,.35,0])
    limiter.measured=np.array([0,.4,0])
    command,safe=checked_contact_turn_command([0,.35,0],[0,0],0,p,[],limiter,.01,target_position=[0,.15])
    assert not safe  # Actual momentum must not be clipped to the intended goal.
    np.testing.assert_allclose(command,[0,0,0])
