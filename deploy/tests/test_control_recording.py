"""Control evidence fidelity and subscription-only recording, without robot I/O."""
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import load, REPO
from deploy.recording.control import ControlLog, control_topics, record_control


def low_message(keys=0):
    remote = [85, 81, keys & 255, keys >> 8]+[0]*36
    data = {'wireless_remote': remote, 'foot_force': [23, 0, 31, 12], 'tick': 31,
            'motor_state': [{'q': np.float32(.3), 'dq': .4, 'tau_est': .5, 'mode': 1} for _ in range(20)]}
    return SimpleNamespace(wireless_remote=remote, data=data)


def test_state_sampling_preserves_short_remote_button_edges_and_numpy_values():
    stream = io.StringIO()
    log = ControlLog(stream, {'/lowstate': ('LowState', 100.)}, 1_000_000_000)
    for ns, keys in [(0, 0), (1_000_000, 0), (2_000_000, 4096), (3_000_000, 0)]:
        log.receive('/lowstate', low_message(keys), lambda m: m.data.copy(), 1_000_000_000+ns)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r['data']['remote_keys'] for r in rows] == [0, 4096, 0]
    assert rows[1]['received_monotonic_ns'] == 1_002_000_000
    assert rows[1]['elapsed_s'] == .002
    assert len(rows[0]['data']['motor_state']) == 12
    assert rows[0]['data']['motor_state'][0]['q'] == pytest.approx(.3)
    assert rows[0]['data']['foot_force'] == [23, 0, 31, 12]
    assert log.stats['/lowstate']['sampled_out'] == 1


def test_rpc_callbacks_never_throttled_and_original_wire_string_is_retained():
    stream = io.StringIO()
    topic = '/api/obstacles_avoid/request'
    log = ControlLog(stream, {topic: ('Request', 0.)}, 0)
    for ns, payload in enumerate(['{"x":0.15,"mode":0}', 'not-json', '{"x":0.25}']):
        msg = SimpleNamespace(data={'header': {'identity': {'api_id': 1003}}, 'parameter': payload, 'binary': [1, 2]})
        log.receive(topic, msg, lambda m: m.data.copy(), ns)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == 3 and log.stats[topic]['sampled_out'] == 0
    assert rows[0]['data']['parameter_json']['x'] == .15
    assert rows[1]['data']['parameter'] == 'not-json'
    assert rows[0]['data']['binary'] == [1, 2]


def test_invalid_sample_and_disk_failure_are_not_hidden():
    log = ControlLog(io.StringIO(), {'topic': ('State', 0.)}, 0)
    log.receive('topic', object(), lambda _: {'speed': float('nan')}, 1)
    assert log.stats['topic']['decode_errors'] == 1
    assert log.stats['topic']['written'] == 0
    log.stream.close()
    with pytest.raises(ValueError):
        log.receive('topic', object(), lambda _: {'speed': .1}, 2)


def test_topic_profile_uses_configured_inputs_and_both_remote_streams():
    robot = dict(load()['robot'], odom_topic='/custom_odom', sport_state_topic='rt/lf/sportmodestate')
    topics = control_topics(robot)
    assert topics['/lf/sportmodestate'][1] == 100.
    assert '/custom_odom' in topics
    assert '/wirelesscontroller' in topics and '/wirelesscontroller_unprocessed' in topics
    assert topics['/api/obstacles_avoid/request'][1] == 0.


@pytest.mark.parametrize('interrupt,missing', [(False, None), (True, None), (False, '/utlidar/robot_odom'),
                                              (False, '/wirelesscontroller_unprocessed')])
def test_standalone_capture_only_subscribes_and_closes_with_honest_summary(tmp_path, monkeypatch, interrupt, missing):
    from deploy.recording import control
    clock = [100.]
    monkeypatch.setattr(control, 'time', SimpleNamespace(monotonic=lambda: clock[0], monotonic_ns=lambda: int(clock[0]*1e9)))
    class Node:
        def __init__(self):
            self.subs, self.destroyed = [], False
        def create_subscription(self, kind, topic, callback, qos):
            self.subs.append((topic, callback))
        def get_publishers_info_by_topic(self, topic):
            return [SimpleNamespace(node_name='source', node_namespace='/', endpoint_gid=[1, 2])]
        def destroy_node(self):
            self.destroyed = True
        # No publisher/client/SDK methods: any attempt to send would fail.
    node = Node()
    running = [False]
    def spin_once(_node, timeout_sec):
        clock[0] += .2
        for topic, callback in node.subs:
            if topic == missing:
                continue
            callback(SimpleNamespace(data={'sample': .1}))
        if interrupt:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    ros = SimpleNamespace(init=lambda: running.__setitem__(0, True), ok=lambda: running[0],
                          shutdown=lambda: running.__setitem__(0, False), create_node=lambda _: node, spin_once=spin_once)
    monkeypatch.setitem(sys.modules, 'rclpy', ros)
    monkeypatch.setitem(sys.modules, 'rclpy.qos', SimpleNamespace(
        QoSProfile=lambda **kw: kw, ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1), HistoryPolicy=SimpleNamespace(KEEP_LAST=1)))
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py.utilities', SimpleNamespace(get_message=lambda name: name))
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py.convert', SimpleNamespace(message_to_ordereddict=lambda m: m.data))
    monkeypatch.setattr(control, 'provenance', lambda _: {'source_commit': 'offline-test'})
    old_signal = signal.getsignal(signal.SIGINT)
    code = record_control(load(), tmp_path, duration=3.)
    directory = next(tmp_path.iterdir())
    summary = json.loads((directory/'control_summary.json').read_text())
    required_missing = missing == '/utlidar/robot_odom'
    assert code == (2 if required_missing else 0)
    assert summary['finished_cleanly'] and summary['error'] is None
    assert summary['transport_completeness_verified'] is False
    assert summary['hardware_validated'] is False
    assert summary['reason'] == ('signal_2' if interrupt else 'duration_elapsed')
    assert summary['missing_feedback_topics'] == ([missing] if required_missing else [])
    assert summary['missing_remote_topics'] == ([missing] if missing == '/wirelesscontroller_unprocessed' else [])
    assert node.destroyed and not running[0]
    assert signal.getsignal(signal.SIGINT) == old_signal
    assert (directory/'control.jsonl').stat().st_size > 0
    assert (directory/'stderr.log').exists()


@pytest.mark.parametrize('args', [['--control','--duration','nan'], ['--duration','5'], ['--control','--wait','0']])
def test_control_cli_rejects_invalid_or_ambiguous_options_before_ros(args):
    process = subprocess.run([sys.executable, str(REPO/'deploy/record.py')]+args, capture_output=True, text=True)
    assert process.returncode == 2
    assert 'error:' in process.stderr
