import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_interaction_motion import (
    GO2_GOAL_TOLERANCE_M,goal_within_tolerance,MeasuredStopGate)


@pytest.mark.parametrize('distance,inside',[(0.,True),(.199,True),(.22008145059532289,True),
    (.249999,True),(.25,False),(.251,False),(-.01,False),(float('nan'),False),(float('inf'),False)])
def test_requested_goal_range_has_one_strict_boundary(distance,inside):
    assert GO2_GOAL_TOLERANCE_M==.25
    assert goal_within_tolerance(distance) is inside


def test_inside_goal_range_still_requires_actual_stopping_and_stability():
    goal=np.array([3.3,1.4]);position=np.array([3.3,1.18,.334])
    assert goal_within_tolerance(np.linalg.norm(position[:2]-goal))
    stop=MeasuredStopGate()
    assert not stop.update(position,[.15,0.,0.],[0.,0.,0.],1.,0.)
    assert not stop.update(position,[0.,0.,0.],[0.,0.,0.],1.,.1)
    assert stop.update(position,[0.,0.,0.],[0.,0.,0.],1.,.51)
    # A settled pose outside the requested bound cannot be counted after drift.
    drifted=np.array([3.3,1.14,.334])
    assert not stop.update(drifted,[0.,0.,0.],[0.,0.,0.],1.,.52)
    assert stop.update(drifted,[0.,0.,0.],[0.,0.,0.],1.,.93)
    assert not goal_within_tolerance(np.linalg.norm(drifted[:2]-goal))
