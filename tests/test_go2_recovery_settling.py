import numpy as np
from grutopia_extension.interactive_navigation.go2_recovery_settling import RecoveryMotionGate


def test_translation_waits_for_turn_to_settle_once():
    gate=RecoveryMotionGate()
    np.testing.assert_array_equal(gate.filter([0,0,.8],[0,0,0],0),[0,0,.8])
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[0,0,.3],.1),[0,0,0])
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[0,0,0],.5),[0,0,0])
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[0,0,0],.71),[-.25,0,0])
    # Once translation starts, ordinary gait oscillation is not another switch.
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[-.2,0,.12],.8),[-.25,0,0])


def test_reverse_recovery_does_not_immediately_reverse_measured_walking():
    gate=RecoveryMotionGate()
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[.3,0,0],0),[0,0,0])
    gate.filter([-.25,0,0],[0,0,0],.5)
    np.testing.assert_array_equal(gate.filter([-.25,0,0],[0,0,0],.71),[-.25,0,0])
    np.testing.assert_array_equal(gate.filter([.25,0,0],[-.2,0,0],.8),[0,0,0])


def test_navigation_handoff_requires_measured_stable_window():
    gate=RecoveryMotionGate()
    assert not gate.leave_recovery([-.2,0,0],0)
    assert not gate.leave_recovery([0,0,0],1)
    assert not gate.leave_recovery([0,0,.11],1.1)
    assert not gate.leave_recovery([0,0,0],1.2)
    assert gate.leave_recovery([0,0,0],1.41)
