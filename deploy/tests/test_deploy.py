import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from deploy.app.config import load, REPO, check_motion_configuration
from deploy.app.tasks import Tasks
from deploy.recording.replay import snapshots
from deploy.robot.localization import camera_cloud, PoseContinuity, rotation, multiply
from deploy.robot.observation import Snapshot, normalize
from deploy.robot.safety import CommandGuard, check_observation, checked_velocity
from deploy.robot.sensors import decode_image
from deploy.robot.motion import _worker
from deploy.tests.make_fixture import make_fixture

def test_unverified_geometry_cannot_enable_motion():
    config = load()
    config['robot']['deskew_and_self_filter_verified'] = False
    with pytest.raises(ValueError, match='deskew_and_self_filter_verified'):
        check_motion_configuration(config)

def test_timeout_latches_and_late_packets_cannot_restart():
    guard = CommandGuard(load()['runtime']['control'])
    guard.arm(10.)
    assert guard.submit([.15, 0, 0], 10., 10.3, 10.)
    assert guard.tick(10.1)[0] > 0
    assert guard.tick(10.4) is None
    assert guard.reason == 'command_timeout'
    assert not guard.submit([.15, 0, 0], 10.5, 10.8, 10.5)
    assert guard.tick(10.6) is None

@pytest.mark.parametrize('command', [[float('nan'), 0, 0], [1, 2], [0, float('inf'), 0]])
def test_invalid_commands_latch_off(command):
    guard = CommandGuard(load()['runtime']['control'])
    guard.arm(1)
    assert not guard.submit(command, 1, 1.2, 1)
    assert not guard.armed

@pytest.mark.parametrize('yaw', [-1.119007271, 1.119007271])
def test_turn_saturation_does_not_shrink_recorded_navigation_translation(yaw):
    # Actual first live navigation request: the old whole-twist scale reduced
    # vx from 0.1117 to 0.0349 solely because yaw exceeded its 0.35 limit.
    requested = np.array([.111664123, -.08, yaw])
    original = requested.copy()
    limited = checked_velocity(requested, load()['runtime']['control'])
    np.testing.assert_allclose(limited[:2], requested[:2])
    assert limited[2] == pytest.approx(np.sign(yaw)*load()['runtime']['control']['max_wz'])
    np.testing.assert_array_equal(requested, original)

@pytest.mark.parametrize('requested', [[1.6, .30, .2], [-.8, .60, -.2], [1.6, -.60, 1.2]])
def test_translation_saturation_preserves_checked_segment_direction(requested):
    limits = load()['runtime']['control']
    output = checked_velocity(requested, limits)
    assert abs(output[0]) <= limits['max_vx']
    assert abs(output[1]) <= limits['max_vy']
    assert abs(output[2]) <= limits['max_wz']
    assert output[0]*requested[1] == pytest.approx(output[1]*requested[0])
    assert np.dot(output[:2], requested[:2]) > 0
    assert np.linalg.norm(output[:2]) <= np.linalg.norm(requested[:2])

@pytest.mark.parametrize('requested', [[0., 0., 0.], [.01, -.02, .03], [0., 0., 1.2]])
def test_velocity_limit_never_invents_translation(requested):
    output = checked_velocity(requested, load()['runtime']['control'])
    np.testing.assert_allclose(output[:2], requested[:2])

def test_turning_forward_command_keeps_acceleration_and_timeout_guards():
    limits = load()['runtime']['control']
    guard = CommandGuard(limits)
    guard.arm(10.)
    previous = np.zeros(3)
    for now in np.arange(10.05, 11.81, .05):
        assert guard.submit([.80, 0., 1.2], now, now+.3, now)
        output = guard.tick(now)
        assert abs(output[0]-previous[0]) <= limits['acceleration']*.05+1e-9
        assert abs(output[2]-previous[2]) <= limits['angular_acceleration']*.05+1e-9
        previous = output
    np.testing.assert_allclose(output, [.80, 0., limits['max_wz']])
    assert guard.tick(now+.31) is None
    assert guard.reason == 'command_timeout' and not guard.armed

def test_camera_geometry_and_invalid_pixel_alignment():
    depth = np.array([[1., np.nan], [2., 2.]], dtype=np.float32)
    k = np.array([[2., 0, 0], [0, 2., 0], [0, 0, 1]])
    points = camera_cloud(depth, k, (1, 2, 3), (1, 0, 0, 0))
    np.testing.assert_allclose(points[0], [1, 2, 4])
    np.testing.assert_allclose(points[3], [2, 3, 5])
    assert np.isnan(points[1]).all()
    # View-frame +X is optical-frame +Z; this must match actual camera viewing yaw.
    np.testing.assert_allclose(rotation(multiply((1, 0, 0, 0), (.5, .5, -.5, .5)))[:, 0], [0, 0, 1])

def test_image_padding_endianness_and_depth_units():
    raw = np.array([1000, 2000, 99, 3000, 4000, 99], dtype='>u2').tobytes()
    message = SimpleNamespace(encoding='16UC1', is_bigendian=True, width=2, height=2, step=6, data=raw)
    np.testing.assert_allclose(decode_image(message), [[1, 2], [3, 4]])

def test_pose_jump_latches_until_new_session():
    continuity = PoseContinuity(.5, .7)
    continuity.check((0, 0, .3), (1, 0, 0, 0))
    with pytest.raises(ValueError, match='Localization jump'):
        continuity.check((2, 0, .3), (1, 0, 0, 0))
    with pytest.raises(ValueError):
        continuity.check((2, 0, .3), (1, 0, 0, 0))

def test_no_ground_truth_leaks_and_stale_data_rejected(tmp_path):
    snapshot = next(snapshots(make_fixture(tmp_path)))
    snapshot.observation['sensors']['camera'] = {
        'rgba': np.zeros((2, 2, 3), dtype=np.uint8), 'depth': np.ones((2, 2)),
        'pointcloud': np.ones((4, 3)), 'position': [0, 0, .4], 'orientation': [1, 0, 0, 0],
        'horizontal_fov_degrees': 90, 'capture_step': 1, 'bounding_box_2d_tight': 'secret_simulator_labels'}
    observation = normalize(snapshot.observation)
    assert 'bounding_box_2d_tight' not in observation['sensors']['camera']
    cfg = load()
    with pytest.raises(ValueError, match='stale'):
        check_observation(snapshot, 1001, cfg['robot'], cfg['runtime']['control'], cfg['navigation']['mapping'])

def test_task_pause_and_switch_preserve_map(tmp_path):
    source = list(snapshots(make_fixture(tmp_path)))
    engine = Tasks(load()['navigation'])
    engine.update(source[0], source[0].captured_at)
    retained = engine.engine.layers
    engine.start('navigate', (1, 0, .3), 1000.)
    assert engine.engine.layers is retained
    engine.update(source[1], 1000.1)
    elapsed = engine.elapsed
    engine.pause()
    engine.update(source[2], 1009.)
    assert engine.elapsed == elapsed
    engine.resume(1010.)
    engine.update(source[3], 1010.1)
    assert engine.elapsed == pytest.approx(elapsed+.1)
    engine.cancel()
    engine.start('navigate', (2, 0, .3), 1011.)
    assert engine.engine.layers is retained

class FakeSport:
    """Only the high-level API exists: tests cannot silently use mode release."""
    def __init__(self, robot, control):
        self.takeover = False
        self.is_stopped = True
        self.log = robot.get('log')
    def healthy(self, now):
        return True
    def stopped(self, now):
        return self.is_stopped
    def stationary_feedback_time(self, now):
        return now if self.stopped(now) else None
    def move(self, velocity):
        if self.log is not None:
            self.log.put('move')
        self.is_stopped = False
    def stop(self):
        if self.log is not None:
            self.log.put('stop')
        self.is_stopped = True

def test_independent_writer_stops_when_parent_produces_no_new_commands():
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, {}, load()['runtime']['control'], FakeSport))
    process.start()
    child.close()
    events = []
    try:
        assert parent.poll(5)
        assert parent.recv()['state'] == 'manual'
        parent.send({'op': 'arm'})
        assert parent.poll(1)
        assert parent.recv()['state'] == 'active'
        now = time.monotonic()
        parent.send({'op': 'velocity', 'velocity': [.1, 0, 0], 'generated_at': now, 'expires_at': now+.2})
        deadline = now+2
        while time.monotonic() < deadline:
            if parent.poll(.05):
                events.append(parent.recv())
                if events[-1]['state'] == 'stopped':
                    break
        assert any(e['state'] == 'stopping' and e['detail'] == 'command_timeout' for e in events)
        assert events[-1]['state'] == 'stopped'
        parent.send({'op': 'close'})
        process.join(2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1)
        parent.close()

def test_actual_cli_replay_runs_existing_planner_without_hardware(tmp_path):
    fixture = make_fixture(tmp_path / 'input')
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, str(REPO/'deploy/run.py'), '--replay', str(fixture),
                             '--task', 'navigate', '--goal', '1', '0', '--output', str(output)],
                            cwd='/tmp', text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    report = next(output.glob('*/run_summary.json'))
    summary = json.loads(report.read_text())
    assert summary['task_result']['success'] is True
    assert summary['sent_to_robot'] is False
    assert 'simulated' not in summary['mapping']['movable_pose_source']
    assert (report.parent/'final_map'/'layered_map.npz').is_file()
    decisions = [json.loads(line) for line in (report.parent/'events.jsonl').read_text().splitlines()]
    assert any(e['requested_velocity'] and e['requested_velocity'][0] > 0 for e in decisions)

@pytest.mark.parametrize('argv, motion_allowed, demo', [
    ([], True, False),
    (['--camera', 'none'], True, False),
    (['--allow-motion'], True, False),
    (['--observe-only'], False, False),
    (['--demo'], False, True),
])
def test_cli_motion_permission_is_enabled_only_for_live_navigation(monkeypatch, argv, motion_allowed, demo):
    from deploy import run
    from deploy.app import runtime

    calls = []

    def live(args, config):
        calls.append(args)
        return 0

    monkeypatch.chdir(REPO/'deploy')
    monkeypatch.setattr(runtime, 'run_live', live)
    assert run.main(argv) == 0
    assert len(calls) == 1
    assert calls[0].allow_motion is motion_allowed
    assert calls[0].demo is demo


def test_replay_and_motion_flags_are_mutually_exclusive(tmp_path):
    result = subprocess.run([sys.executable, str(REPO/'deploy/run.py'), '--replay', str(tmp_path),
                             '--allow-motion'], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'not allowed with argument' in result.stderr

def test_unarmed_writer_does_not_interfere_with_manual_control():
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    log = context.Queue()
    process = context.Process(target=_worker, args=(child, {'log': log}, load()['runtime']['control'], FakeSport))
    process.start()
    child.close()
    try:
        assert parent.poll(5)
        assert parent.recv()['state'] == 'manual'
        parent.send({'op': 'close'})
        process.join(2)
        assert process.exitcode == 0
        import queue
        with pytest.raises(queue.Empty):
            log.get(timeout=.1)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1)
        parent.close()
        log.close()

def test_navigation_imports_do_not_load_simulator_modules():
    code = "import sys; from deploy.app.tasks import Tasks; assert not any(k == 'omni' or k.startswith('omni.') or k == 'isaacsim' for k in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
