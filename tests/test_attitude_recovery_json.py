"""Plateau decisions must preserve their boolean meaning in JSON telemetry."""
import json,math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery


def quaternion(degrees):
    angle=math.radians(degrees)/2
    return np.array([math.cos(angle),0.,math.sin(angle),0.])


@pytest.mark.parametrize('varying',[False,True])
def test_plateau_true_and_false_are_native_json_booleans(varying):
    recovery=NavigationAttitudeRecovery(minimum_hold_ticks=6,stable_ticks=3,maximum_hold_ticks=20)
    position=np.array([0.,0.,.42]);recovery.update(position,quaternion(20.))
    for index in range(6):
        detail=recovery.update(position,quaternion(12.+(2.*(index%2) if varying else 0.)))
        saved=json.loads(json.dumps(detail))
        assert type(detail['settled']) is bool and saved['settled'] is detail['settled']
    assert detail['state']==('hold' if varying else 'released')
    assert detail['settled'] is (not varying)
