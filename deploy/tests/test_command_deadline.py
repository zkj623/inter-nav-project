"""Packet arrival cannot bypass a deadline when the worker has not ticked yet."""
import numpy as np
import pytest

from deploy.app.config import load
from deploy.robot.safety import CommandGuard


@pytest.fixture
def guard():
    return CommandGuard(load()['runtime']['control'])


@pytest.mark.parametrize('late_by', [0., .01, 1.])
@pytest.mark.parametrize('prior_command', [False, True])
def test_fresh_packet_at_or_after_old_expiry_latches_without_tick(guard, late_by, prior_command):
    guard.arm(10.)
    if prior_command:
        assert guard.submit([.2, 0., 0.], 10., 10.25, 10.)
        assert guard.tick(10.05)[0] > 0.
    old_expiry = guard.expires
    now = old_expiry+late_by

    # No tick observes the deadline before this otherwise-valid new packet.
    assert not guard.submit([.2, 0., 0.], now, now+.3, now)
    assert not guard.armed and guard.reason == 'command_timeout'
    assert guard.expires == old_expiry
    np.testing.assert_array_equal(guard.output, np.zeros(3))
    assert not guard.submit([.2, 0., 0.], now+.01, now+.31, now+.01)
    assert guard.tick(now+.02) is None
    assert guard.reason == 'command_timeout'


def test_packet_before_old_expiry_can_renew_and_then_expire_normally(guard):
    guard.arm(10.)
    original_expiry = guard.expires
    now = original_expiry-1e-6
    assert guard.submit([.2, 0., .1], now, now+.3, now)
    assert guard.armed and guard.expires == now+.3
    assert guard.tick(original_expiry+.01) is not None
    assert guard.tick(guard.expires) is None
    assert guard.reason == 'command_timeout'


@pytest.mark.parametrize('late_by', [0., .01, 1.])
def test_fresh_packet_cannot_cross_trial_deadline_without_tick(guard, late_by):
    guard.arm(10., duration_s=.5)
    assert guard.submit([.2, 0., 0.], 10.2, 10.5, 10.2)
    assert guard.submit([.2, 0., 0.], 10.4, 10.7, 10.4)
    assert guard.tick(10.45)[0] > 0.
    now = 10.5+late_by

    assert not guard.submit([.2, 0., 0.], now, now+.3, now)
    assert not guard.armed and guard.reason == 'test_duration_elapsed'
    assert guard.deadline == 10.5 and guard.expires == 10.7
    np.testing.assert_array_equal(guard.output, np.zeros(3))
    assert not guard.submit([.2, 0., 0.], now+.01, now+.31, now+.01)
    assert guard.tick(now+.02) is None
    assert guard.reason == 'test_duration_elapsed'


def test_trial_deadline_stays_fixed_during_timely_command_renewal(guard):
    guard.arm(10., duration_s=.5)
    for now in (10.1, 10.2, 10.3, 10.4, 10.49):
        assert guard.submit([.2, 0., 0.], now, now+.3, now)
        assert guard.tick(now) is not None
        assert guard.deadline == 10.5
    assert guard.tick(10.5) is None
    assert guard.reason == 'test_duration_elapsed'


def test_only_explicit_arm_can_create_new_lease_after_late_packet(guard):
    guard.arm(10.)
    assert not guard.submit([.2, 0., 0.], 11., 11.3, 11.)
    guard.arm(12.)
    assert guard.submit([.2, 0., 0.], 12., 12.3, 12.)
    assert guard.tick(12.1) is not None
    assert guard.reason == 'active'
