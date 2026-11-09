"""Real display-process integration; blocking/failure must not block navigation."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import time

import numpy as np
import pytest

from deploy.app.config import load, REPO
from deploy.app.session import Session, decode_packet
from deploy.app.tasks import Tasks
from deploy.robot.observation import Snapshot


def wait_until(predicate, timeout=8.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('display process did not reach the expected state')


@pytest.fixture
def detached_scene():
    config = load()
    tasks = Tasks(config['navigation'])
    session = Session(config, REPO, port=0, demo=True, separate_cloud=True)
    try:
        yield session, tasks
    finally:
        if session.renderer.is_alive() and hasattr(signal, 'SIGCONT'):
            os.kill(session.renderer.pid, signal.SIGCONT)
        session.close({'completed': True}, grace=0)
        if session.renderer.is_alive():
            session.renderer.kill()
            session.renderer.join(timeout=2)


def snapshot(step, points):
    now = time.monotonic()
    return Snapshot({'position': np.array([0., 0., .3]),
                     'orientation': np.array([1., 0., 0., 0.]),
                     'sensors': {'lidar': {'pointcloud': np.asarray(points, dtype=np.float32),
                                           'position': np.array([.05, .05, .55]),
                                           'physics_step': step, 'max_range': 8.}}},
                    now, now, now, None, (0., 0., 0.), step,
                    geometry_calibrated=True, planning_frame='experiment')


def update(session, tasks, packet, *, publish=False, accepted=True):
    if accepted:
        # Exercise every accepted scan without real sleeps; the release mapper
        # intentionally skips inputs inside its local map scheduling period.
        now = max(packet.captured_at, tasks.engine.layers.last_local + .101)
        tasks.update(packet, now)
    session.last_publish = float('-inf') if publish else time.monotonic()+60.
    session.update(packet, tasks, 'disabled')


def published_scene(session, expected):
    wait_until(lambda: session.frame_id == expected)
    return decode_packet(session.batch(0, 0)['frames'][-1]['packet'])[1]


def contains(points, point):
    return np.any(np.linalg.norm(np.asarray(points)-point, axis=1) < .01)


def test_detached_cloud_fuses_nonpublished_scans_but_not_repeats_or_rejected_input(detached_scene, monkeypatch):
    session, tasks = detached_scene
    import deploy.visualization.scene as scene_module
    def must_not_extract():
        raise AssertionError('navigation thread extracted cumulative display points')
    monkeypatch.setattr(scene_module, 'display_points', must_not_extract)
    old, unseen, farther = [1.05, .05, .55], [1.05, 1.05, .55], [3.05, .05, .55]
    update(session, tasks, snapshot(1, [old, unseen]), publish=True)
    published_scene(session, 1)
    update(session, tasks, snapshot(2, [farther]))  # Fusion without a published frame.
    for _ in range(4):
        update(session, tasks, snapshot(2, [farther]))
    update(session, tasks, snapshot(2, [farther]), publish=True)
    second = published_scene(session, 2)
    assert contains(second['points'], old)  # Repeated reads cannot count as new free rays.
    update(session, tasks, snapshot(3, [farther]))
    update(session, tasks, snapshot(4, [farther]), publish=True)
    fourth = published_scene(session, 3)
    assert contains(fourth['points'], old)  # Historical display, independent of navigation clearing.
    assert contains(fourth['points'], unseen) and contains(fourth['points'], farther)
    cloud_at = fourth['cloud_updated_at']
    rejected_point = [5.05, 1.05, .55]
    update(session, tasks, snapshot(5, [rejected_point]), publish=True, accepted=False)
    rejected = published_scene(session, 4)
    assert not contains(rejected['points'], rejected_point)
    assert rejected['cloud_physics_step'] == 4 and rejected['cloud_updated_at'] == cloud_at
    assert 'points' not in session.capture.capture(snapshot(6, [farther]), tasks)
    assert session.cloud_input_dropped == session.dropped == 0
    session.close({'completed': True}, grace=0)
    assert session.summary['telemetry_complete'] and session.summary['cloud_history_complete']


def pause_renderer(session):
    os.kill(session.renderer.pid, signal.SIGSTOP)
    def stopped():
        status = Path(f'/proc/{session.renderer.pid}/status').read_text()
        return any(line.startswith('State:') and 'T' in line for line in status.splitlines())
    wait_until(stopped)


@pytest.mark.skipif(not hasattr(signal, 'SIGSTOP') or not Path('/proc').exists(), reason='Linux process control')
def test_full_display_queue_drops_scans_and_frames_separately_without_blocking(detached_scene):
    session, tasks = detached_scene
    update(session, tasks, snapshot(1, [[1.05, .05, .55]]), publish=True)
    published_scene(session, 1)
    pause_renderer(session)
    def send_scans(start, publish):
        for step in range(start, start+10):
            update(session, tasks, snapshot(step, [[2.05, .05, .55]]), publish=publish)
    try:
        # A bounded timeout would fail if put() waited for the stopped child.
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                executor.submit(send_scans, 2, False).result(timeout=3)
                assert session.cloud_input_dropped > 0
                assert session.dropped == 0  # These inputs did not request display frames.
                executor.submit(send_scans, 12, True).result(timeout=3)
                assert session.dropped > 0
                assert session.cloud_input_dropped > session.dropped
                assert tasks.state == 'observe' and tasks.map_ready
            finally:
                os.kill(session.renderer.pid, signal.SIGCONT)
    finally:
        if session.renderer.is_alive():
            os.kill(session.renderer.pid, signal.SIGCONT)
    session.close({'completed': True}, grace=0)
    assert session.summary['telemetry_complete']
    assert not session.summary['cloud_history_complete']


def test_exited_renderer_is_reported_without_faulting_navigation(detached_scene):
    session, tasks = detached_scene
    session.renderer.kill()
    session.renderer.join(timeout=2)
    wait_until(lambda: session.render_failure)
    update(session, tasks, snapshot(1, [[1.05, .05, .55]]))
    assert tasks.state == 'observe' and tasks.map_ready
    assert session.cloud_input_dropped == 1 and session.dropped == 0
    assert any(e['event'] == 'visualization_error' for e in session.batch(0, 0)['events'])
    session.close({'completed': True}, grace=0, drain_timeout=.1)
    assert not session.summary['telemetry_complete']
    assert not session.renderer.is_alive() and not session.worker.is_alive()


@pytest.mark.skipif(not hasattr(signal, 'SIGSTOP') or not Path('/proc').exists(), reason='Linux process control')
def test_close_kills_stopped_renderer_and_freezes_recording_boundary(detached_scene):
    session, tasks = detached_scene
    update(session, tasks, snapshot(1, [[1.05, .05, .55]]), publish=True)
    published_scene(session, 1)
    pause_renderer(session)
    next_snapshot = snapshot(2, [[2.05, .05, .55]])
    update(session, tasks, next_snapshot, publish=True)
    started = time.monotonic()
    session.close({'completed': True}, grace=0, drain_timeout=.05)
    assert time.monotonic()-started < 5.
    assert not session.renderer.is_alive() and not session.worker.is_alive()
    final = session.batch(0, 0)
    assert not final['summary']['telemetry_complete']
    assert not final['summary']['completed'] and not final['summary']['cloud_history_complete']
    assert final['summary']['final_frame_id'] == 1 and final['producer_dropped'] == 1
    assert final['summary']['final_event_id'] == final['events'][-1]['event_id']
    session.event('late_event')
    session.update(next_snapshot, tasks, 'disabled')
    assert session.batch(0, 0) == final
