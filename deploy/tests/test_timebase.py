"""Clock mapping with synthetic clocks only; no ROS or robot clock writes."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from deploy.app.config import load
from deploy.robot.timebase import SourceTimeBridge
from deploy.robot.sensors import RosSensors


EPOCH_NS = 1_800_000_000_000_000_123
MONO_NS = 10_000_000_000


def test_receipt_delay_never_changes_capture_time_or_initializes_offset():
    bridge = SourceTimeBridge(EPOCH_NS, MONO_NS)
    captures = [bridge.captured_at(EPOCH_NS, MONO_NS + delay, .4)
                for delay in (10_000_000, 300_000_000)]
    assert captures == [10., 10.]
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(EPOCH_NS, MONO_NS + 410_000_000, .4)


def test_first_packet_can_be_stale_and_common_old_epoch_is_not_auto_repaired():
    bridge = SourceTimeBridge(EPOCH_NS, MONO_NS)
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(EPOCH_NS - 1_000_000_000, MONO_NS, .4)
    wrong_boot_clock = SourceTimeBridge(5_000_000_000, MONO_NS)
    with pytest.raises(ValueError, match='clock mismatch'):
        wrong_boot_clock.captured_at(EPOCH_NS, MONO_NS, .4)


def test_anchor_sampling_uncertainty_is_retained_as_age():
    ticks = iter([MONO_NS, MONO_NS + 2_000_000])
    bridge = SourceTimeBridge.from_clock(lambda: EPOCH_NS, monotonic_now_ns=lambda: next(ticks))
    assert bridge.captured_at(EPOCH_NS, MONO_NS + 200_000_000, .4) == 10.
    assert bridge.metadata['anchor_uncertainty_ns'] == 2_000_000
    assert bridge.metadata['receipt_offset_estimation'] is False
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(EPOCH_NS, MONO_NS + 401_000_000, .4)


@pytest.mark.parametrize('span', [-1, 10_000_001])
def test_interrupted_anchor_sampling_is_rejected(span):
    ticks = iter([MONO_NS, MONO_NS + span])
    with pytest.raises(ValueError, match='sampling was interrupted'):
        SourceTimeBridge.from_clock(lambda: EPOCH_NS, monotonic_now_ns=lambda: next(ticks))


def test_epoch_integer_precision_and_shared_mapping_preserve_pairing_difference():
    bridge = SourceTimeBridge(EPOCH_NS, MONO_NS)
    # Differences smaller than a float's resolution at a Unix epoch survive.
    a = bridge.captured_at(EPOCH_NS + 1, MONO_NS + 100_000_000, .4)
    b = bridge.captured_at(EPOCH_NS + 21_000_001, MONO_NS + 100_000_000, .4)
    assert a == pytest.approx(10.000000001, abs=1e-12)
    assert b - a == pytest.approx(.021, abs=1e-12)


def test_tolerated_future_skew_does_not_extend_capture_past_receipt():
    bridge = SourceTimeBridge(EPOCH_NS, MONO_NS)
    assert bridge.captured_at(EPOCH_NS + 40_000_000, MONO_NS, .4) == 10.
    with pytest.raises(ValueError, match='clock mismatch'):
        bridge.captured_at(EPOCH_NS + 60_000_000, MONO_NS, .4)


def test_source_drift_or_jump_is_not_followed_to_hide_bad_ages():
    bridge = SourceTimeBridge(EPOCH_NS, MONO_NS)
    # Both clocks advance 100 seconds, but the source jumps/drifts ahead.
    with pytest.raises(ValueError, match='clock mismatch'):
        bridge.captured_at(EPOCH_NS + 100_100_000_000, MONO_NS + 100_000_000_000, .4)
    with pytest.raises(ValueError, match='stale'):
        bridge.captured_at(EPOCH_NS + 99_500_000_000, MONO_NS + 100_000_000_000, .4)
    assert bridge.metadata['source_anchor_ns'] == EPOCH_NS


def fake_ros(monkeypatch, *, sim_time=False, silent=False):
    state = SimpleNamespace(wall_ns=2_000_000_000, clock_reads=0, destroyed=False,
                            shutdown=False, mono_ns=MONO_NS, subscriptions=[], removed=[])
    def read_clock():
        state.clock_reads += 1
        return SimpleNamespace(nanoseconds=state.wall_ns)
    def subscribe(cls, topic, callback, qos):
        subscription = (topic, callback)
        state.subscriptions.append(subscription)
        return subscription
    def remove(subscription):
        state.subscriptions.remove(subscription)
        state.removed.append(subscription)
    def spin_once(node, timeout_sec):
        state.mono_ns += 10_000_000
        if not silent:
            for _, callback in list(state.subscriptions):
                callback(message(EPOCH_NS + state.mono_ns - MONO_NS))
    node = SimpleNamespace(
        get_parameter=lambda _: SimpleNamespace(value=sim_time),
        get_clock=lambda: SimpleNamespace(now=read_clock),
        create_subscription=subscribe, destroy_subscription=remove,
        count_publishers=lambda _: 1,
        destroy_node=lambda: setattr(state, 'destroyed', True))
    definitions = {
        'rclpy': dict(init=lambda: None, create_node=lambda _: node, spin=lambda _: None,
                      spin_once=spin_once, shutdown=lambda: setattr(state, 'shutdown', True)),
        'rclpy.qos': dict(qos_profile_sensor_data=None),
        'rclpy.time': dict(Time=object),
        'sensor_msgs.msg': dict(PointCloud2=object, Image=object, CameraInfo=object),
        'nav_msgs.msg': dict(Odometry=object),
        'tf2_ros': dict(Buffer=lambda: None, TransformListener=lambda *args: None),
    }
    for name, attributes in definitions.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr('deploy.robot.sensors.time.monotonic_ns', lambda: state.mono_ns)
    monkeypatch.setattr('deploy.robot.sensors.time.monotonic', lambda: state.mono_ns*1e-9)
    return state


def message(stamp_ns):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def test_sensor_bootstraps_with_wrong_wall_date_and_keeps_fixed_reference(monkeypatch):
    state = fake_ros(monkeypatch)
    sensors = RosSensors(load()['robot'])
    try:
        stamp = EPOCH_NS + state.mono_ns - MONO_NS
        expected = state.mono_ns*1e-9 - .05
        before = sensors._stamp(message(stamp))
        state.wall_ns = EPOCH_NS  # Independent system-date correction after bootstrap.
        state.mono_ns += 100_000_000
        after = sensors._stamp(message(stamp))
        assert before == after
        assert before[1] == pytest.approx(expected)
        assert state.clock_reads == 0  # Neither startup nor callbacks consult ROS/wall date.
        assert len(state.removed) == 2
        assert sensors.timebase.metadata['receipt_offset_estimation'] is True
        assert sensors._stamp(message(stamp + 20_000_000))[1] == pytest.approx(expected + .02)
    finally:
        sensors.close()


def test_sensor_delayed_message_after_bootstrap_is_rejected(monkeypatch):
    state = fake_ros(monkeypatch)
    sensors = RosSensors(load()['robot'])
    try:
        current = EPOCH_NS + state.mono_ns - MONO_NS
        with pytest.raises(ValueError, match='stale'):
            sensors._stamp(message(current - 1_000_000_000))
    finally:
        sensors.close()


@pytest.mark.parametrize('sim_time,silent', [(True, False), (False, True)])
def test_invalid_bootstrap_releases_ros_resources(monkeypatch, sim_time, silent):
    state = fake_ros(monkeypatch, sim_time=sim_time, silent=silent)
    with pytest.raises(ValueError):
        RosSensors(load()['robot'])
    assert state.destroyed and state.shutdown
    assert not state.subscriptions
