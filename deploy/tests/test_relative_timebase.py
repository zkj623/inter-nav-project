"""Relative bootstrap behavior, including its unobservable baseline delay."""
from copy import deepcopy

import pytest

from deploy.robot.timebase import SourceTimeBridge


SOURCE_NS = 1_800_000_000_000_000_123
MONO_NS = 10_000_000_000


def samples():
    # 3 seconds of observations. LiDAR has 60 ms additional pipeline delay.
    return {
        'odom': [(MONO_NS+i*100_000_000, SOURCE_NS+i*100_000_000) for i in range(31)],
        'lidar': [(MONO_NS+i*200_000_000+60_000_000, SOURCE_NS+i*200_000_000)
                  for i in range(16)],
    }


def test_relative_bootstrap_works_without_wall_epoch_and_retains_reserve():
    bridge = SourceTimeBridge.from_samples(samples())
    assert bridge.captured_at(SOURCE_NS, MONO_NS, .4) == pytest.approx(9.95)
    assert bridge.captured_at(SOURCE_NS, MONO_NS+60_000_000, .4) == pytest.approx(9.95)
    metadata = bridge.metadata
    assert metadata['receipt_offset_estimation']
    assert metadata['unknown_baseline_delay'] and not metadata['absolute_age_verified']
    assert metadata['delay_reserve_s'] == .05
    assert not metadata['delay_reserve_is_measured_bound']
    assert metadata['bootstrap_sample_counts'] == {'odom': 31, 'lidar': 16}


def test_source_epoch_need_not_be_unix_wall_date():
    values = samples()
    shifted = {kind: [(r, s-SOURCE_NS+1_000_000_000) for r, s in entries]
               for kind, entries in values.items()}
    bridge = SourceTimeBridge.from_samples(shifted)
    assert bridge.captured_at(1_000_000_000, MONO_NS, .4) == pytest.approx(9.95)


def test_fixed_mapping_rejects_later_delay_and_does_not_absorb_a_pause():
    bridge = SourceTimeBridge.from_samples(samples())
    metadata = bridge.metadata
    stamp = SOURCE_NS+4_000_000_000
    first = bridge.captured_at(stamp, MONO_NS+4_010_000_000, .4)
    delayed = bridge.captured_at(stamp, MONO_NS+4_300_000_000, .4)
    assert first == delayed
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(stamp, MONO_NS+4_360_000_000, .4)
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(stamp, MONO_NS+6_000_000_000, .4)
    assert bridge.metadata == metadata


@pytest.mark.parametrize('change,match', [
    ('missing', 'both odom and lidar'), ('few', 'insufficient'),
    ('short', 'too short'), ('repeated', 'non-increasing'),
    ('jump', 'offset spread'), ('gap', 'receive gap'),
    ('one_old_start', 'offset spread'),
    ('stopped_lidar', 'stream stopped'), ('old_lidar', 'stale'),
    ('future_lidar', 'clock mismatch'),
])
def test_bad_bootstrap_windows_are_rejected(change, match):
    values = deepcopy(samples())
    if change == 'missing':
        del values['lidar']
    elif change == 'few':
        values['odom'] = values['odom'][:19]
    elif change == 'short':
        values['odom'] = values['odom'][:25]
    elif change == 'repeated':
        values['odom'][4] = (values['odom'][4][0], values['odom'][3][1])
    elif change == 'jump':
        values['odom'][15:] = [(r, s+100_000_000) for r, s in values['odom'][15:]]
    elif change == 'one_old_start':
        r, s = values['odom'][0]
        values['odom'][0] = (r, s-2_000_000_000)
    elif change == 'gap':
        values['odom'] = values['odom'][:5] + values['odom'][10:]
    elif change == 'stopped_lidar':
        values['lidar'] = values['lidar'][:2]
    elif change == 'old_lidar':
        values['lidar'] = [(r, s-400_000_000) for r, s in values['lidar']]
    elif change == 'future_lidar':
        values['lidar'] = [(r, s+500_000_000) for r, s in values['lidar']]
    with pytest.raises(ValueError, match=match):
        SourceTimeBridge.from_samples(values)


def test_common_constant_backlog_is_unidentifiable_and_never_certified():
    fresh = samples()
    old = {kind: [(r, s-2_000_000_000) for r, s in entries]
           for kind, entries in fresh.items()}
    before = SourceTimeBridge.from_samples(fresh)
    after = SourceTimeBridge.from_samples(old)
    # One-way samples cannot distinguish 2-second common backlog from a clock
    # offset. This test documents that limit instead of asserting false safety.
    assert before.captured_at(SOURCE_NS, MONO_NS, .4) == after.captured_at(
        SOURCE_NS-2_000_000_000, MONO_NS, .4)
    assert after.metadata['unknown_baseline_delay']
    assert after.metadata['absolute_age_verified'] is False


def test_startup_source_jump_cannot_be_repaired_by_later_receipts():
    bridge = SourceTimeBridge.from_samples(samples())
    with pytest.raises(ValueError, match='clock mismatch'):
        bridge.captured_at(SOURCE_NS+5_000_000_000, MONO_NS+4_000_000_000, .4)
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(SOURCE_NS+3_000_000_000, MONO_NS+4_000_000_000, .4)


def test_age_reserve_does_not_widen_future_tolerance():
    bridge = SourceTimeBridge.from_samples(samples())
    with pytest.raises(ValueError, match='clock mismatch'):
        bridge.captured_at(SOURCE_NS+4_060_000_000, MONO_NS+4_000_000_000, .4)
    assert bridge.captured_at(SOURCE_NS+4_040_000_000, MONO_NS+4_000_000_000, .4) <= 14.
