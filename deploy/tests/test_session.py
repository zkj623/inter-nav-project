"""Single-owner map feed, bounded loss accounting, and real entrypoint integration."""
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import requests

from deploy.app.config import load, REPO
from deploy.app.demo import DemoSensors
from deploy.app.session import Session, decode_packet
from deploy.app.tasks import Tasks
from deploy.recording.replay import snapshots
from deploy.visualization.scene import SceneCapture, render_map, display_points
from grutopia_extension.interactive_navigation.mapping import VoxelPointCloudMap, MappingConfig


def wait_until(callback, timeout=8):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        result = callback()
        if result:
            return result
        time.sleep(.05)
    raise AssertionError('condition did not become true')


@pytest.fixture
def session():
    obj = Session(load(), REPO, port=0, demo=True, frame_limit=2, event_limit=3)
    try:
        yield obj
    finally:
        obj.close({'completed': True}, grace=0)


def test_scene_uses_actual_masks_and_does_not_modify_planner(session):
    tasks = Tasks(load()['navigation'])
    snapshot = DemoSensors(session).latest()
    tasks.update(snapshot, snapshot.captured_at)
    grid = tasks.component.mapping.map.occupancy
    revision = grid.revision
    scene = SceneCapture().capture(snapshot, tasks)
    assert scene['ready']
    np.testing.assert_array_equal(scene['observed'], grid.observed)
    np.testing.assert_array_equal(scene['occupied'], grid.occupied_mask())
    assert grid.revision == revision
    grid.observed[:] = False
    assert scene['observed'].any()  # detached copy, not a moving rendering target
    image = render_map(scene)
    assert image.shape == (720, 720, 3)


def test_unverified_geometry_does_not_invent_map(session):
    tasks = Tasks(load()['navigation'])
    snapshot = DemoSensors(session).latest()
    snapshot.geometry_calibrated = False
    tasks.update(snapshot, snapshot.captured_at)
    session.update(snapshot, tasks, 'disabled')
    wait_until(lambda: session.frame_id)
    data = session.display()
    assert not data['map']['ready']
    assert data['cloud']['kind'] == '当前原始点云（未建图）'
    assert data['status']['motion'] == 'disabled'


def test_packets_keep_raw_observations_and_count_evicted_frames(session):
    tasks = Tasks(load()['navigation'])
    source = DemoSensors(session)
    for i in range(4):
        snapshot = source.latest()
        tasks.update(snapshot, snapshot.captured_at)
        session.last_publish = float('-inf')
        session.update(snapshot, tasks, 'disabled')
        wait_until(lambda: session.frame_id == i+1)
        session.event('test', i=i)
    batch = session.batch(0, 0)
    assert batch['frame_gap'] == 2 and batch['event_gap'] == 2
    restored, scene = decode_packet(batch['frames'][-1]['packet'])
    np.testing.assert_array_equal(restored.observation['sensors']['lidar']['pointcloud'], snapshot.observation['sensors']['lidar']['pointcloud'])
    assert restored.geometry_calibrated and scene['ready']
    assert session.display()['session_id'] == batch['info']['session_id']
    # Every network endpoint is read-only.
    assert requests.post(session.url+'/api/record').status_code == 501
    assert requests.get(session.url+'/api/record?frame=-1').status_code == 400


def test_display_does_not_compress_recording_and_keeps_true_map_age(session, monkeypatch):
    import deploy.app.session as module
    calls = []
    original = module.encode_packet
    def encode(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, 'encode_packet', encode)
    tasks = Tasks(load()['navigation'])
    snapshot = DemoSensors(session).latest()
    tasks.update(snapshot, snapshot.captured_at)
    tasks.last_mapping_at = time.monotonic()-2
    session.update(snapshot, tasks, 'disabled')
    wait_until(lambda: session.frame_id)
    display = session.display()
    assert not calls
    assert display['map']['stale'] and display['cloud']['stale']
    assert display['map']['age'] >= 2
    assert display['performance']['encode_ms'] >= 0 and display['fps'] == 5
    batch = session.batch(0, 0)
    assert len(calls) == 1 and 'packet' in batch['frames'][0]
    assert not any(k.startswith('_') for k in batch['frames'][0])
    session.batch(0, 0)
    assert len(calls) == 1  # compressed packet reused by another recorder


def test_display_sampling_uses_original_public_mapping_api():
    voxels = VoxelPointCloudMap(MappingConfig(voxel_size=.01))
    points = np.column_stack((np.arange(100)/10, np.zeros(100), np.ones(100)))
    voxels.update(points, 1, 'lidar')
    assert len(voxels.points()) == 100
    displayed = display_points(voxels, limit=7)
    assert 1 <= len(displayed) <= 7
    original = voxels.points().copy()
    displayed[:] = 0
    np.testing.assert_array_equal(voxels.points(), original)
    assert display_points(VoxelPointCloudMap(MappingConfig())).shape == (0, 3)
    with pytest.raises(ValueError):
        display_points(voxels, limit=0)


def test_display_sampling_does_not_replace_history_when_count_crosses_budget():
    voxels = VoxelPointCloudMap(MappingConfig(voxel_size=.01))
    voxels.update(np.column_stack((np.arange(20)/10, np.zeros(20), np.ones(20))), 1, 'lidar')
    before = display_points(voxels, limit=10)
    voxels.update(np.array([[5., 0., 1.]]), 2, 'lidar')
    after = display_points(voxels, limit=10)
    # Only the new voxel can replace one sampled old voxel, not a global stride change.
    retained = sum(np.any(np.all(after == p, axis=1)) for p in before)
    assert retained >= 9


def test_session_cloud_updates_vacated_geometry_without_erasing_unseen_history(session):
    tasks = Tasks(load()['navigation'])
    source = DemoSensors(session)
    old = [1.05, .05, .55]
    unseen = [1.05, 1.05, .55]
    farther = [3.05, .05, .55]
    for i, points in enumerate(([old, unseen], [farther], [farther], [farther])):
        snapshot = source.latest()
        snapshot.observation['sensors']['lidar'].update(
            position=np.array([.05, .05, .55]), pointcloud=np.asarray(points), physics_step=i)
        tasks.update(snapshot, snapshot.captured_at)
        session.last_publish = float('-inf')
        session.update(snapshot, tasks, 'disabled')
        wait_until(lambda: session.frame_id == i+1)
    displayed = np.asarray(session.display()['cloud']['points'])
    assert session.display()['cloud']['accumulated']
    assert not np.any(np.linalg.norm(displayed-old, axis=1) < .01)
    for kept in (unseen, farther):
        assert np.any(np.linalg.norm(displayed-kept, axis=1) < .01)
    # Re-reading from another viewer does not mutate or reset the shared map.
    np.testing.assert_array_equal(session.display()['cloud']['points'], displayed)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_real_run_view_record_entrypoints_and_replay(tmp_path):
    source_port, view_port = free_port(), free_port()
    source = f'http://127.0.0.1:{source_port}'
    processes = []

    def start(name, *args):
        process = subprocess.Popen([sys.executable, str(REPO/'deploy'/name), *args], cwd=tmp_path,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        processes.append(process)
        return process

    def frame():
        try:
            response = requests.get(f'http://127.0.0.1:{view_port}/api/frame', timeout=.5)
            if response.status_code == 200 and response.json().get('map', {}).get('ready'):
                return response.json()
        except requests.RequestException:
            pass

    try:
        recorder = start('record.py', '--source', source, '--output', str(tmp_path/'recordings'))
        run = start('run.py', '--demo', '--session-port', str(source_port))
        viewer = start('view.py', '--source', source, '--port', str(view_port))
        data = wait_until(frame, 15)
        assert data['demo'] and data['map']['ready'] and data['rgb']['jpeg']
        assert data['cloud']['kind'] == '累计建图点云'
        assert data['cloud']['frame'] == data['map']['frame'] == 'experiment'
        wait_until(lambda: list((tmp_path/'recordings').glob('*/observations/*.npz')))
        # Closing the viewer is independent of navigation and recording.
        viewer.send_signal(signal.SIGINT)
        output, _ = viewer.communicate(timeout=5)
        assert viewer.returncode == 0, output
        assert run.poll() is None and recorder.poll() is None
        run.send_signal(signal.SIGINT)
        output, _ = run.communicate(timeout=10)
        assert run.returncode == 0, output
        output, _ = recorder.communicate(timeout=10)
        assert recorder.returncode == 0, output
        directory = next((tmp_path/'recordings').iterdir())
        summary = json.loads((directory/'run_summary.json').read_text())
        assert summary['run_final_received'] and summary['session_id'] == data['session_id']
        assert summary['missing_frames'] == summary['missing_events'] == 0
        assert list(directory.glob('frames/*.jpg')) and list(directory.glob('frames/*.png'))
        recorded = list(snapshots(directory))
        assert recorded and all(s.planning_frame == 'experiment' for s in recorded)
        assert recorded[0].coordinate_transform['source_frame'] == 'demo_odom'
        assert summary['run_summary']['coordinate_transform'] == recorded[0].coordinate_transform
        replay = subprocess.run([sys.executable, str(REPO/'deploy/run.py'), '--replay', str(directory),
                                 '--output', str(tmp_path/'replay')], capture_output=True, text=True, timeout=15)
        assert replay.returncode == 0, replay.stdout+replay.stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)


@pytest.mark.parametrize('action', ['operator_stop', 'new_session'])
def test_record_stops_without_affecting_live_session(session, tmp_path, action):
    tasks = Tasks(load()['navigation'])
    snapshot = DemoSensors(session).latest()
    tasks.update(snapshot, snapshot.captured_at)
    session.update(snapshot, tasks, 'disabled')
    wait_until(lambda: session.frame_id)
    process = subprocess.Popen([sys.executable, str(REPO/'deploy/record.py'), '--source', session.url,
                                '--output', str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_until(lambda: list(tmp_path.glob('*/observations/*.npz')))
        if action == 'operator_stop':
            process.send_signal(signal.SIGINT)
        else:
            # Simulate replacing the source with a new run under the same URL.
            with session.lock:
                session.info = dict(session.info, session_id='another_run')
        output, _ = process.communicate(timeout=5)
        assert process.returncode == (0 if action == 'operator_stop' else 2), output
        assert session.summary is None
        assert requests.get(session.url+'/api/frame').status_code == 200
        summary = json.loads(next(tmp_path.glob('*/run_summary.json')).read_text())
        assert not summary['run_final_received'] and not summary['recording_complete']
        assert summary['reason'] == ('operator_stop' if action == 'operator_stop' else 'record_error')
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


def test_demo_refuses_motion_before_initializing_hardware():
    result = subprocess.run([sys.executable, str(REPO/'deploy/run.py'), '--demo', '--allow-motion'], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--demo cannot be combined' in result.stderr


def test_moving_scans_keep_history_in_world_frame_and_session(session):
    tasks = Tasks(load()['navigation'])
    source = DemoSensors(session)
    regions = [np.array([[1., 0., .5], [1., .2, .5]]),
               np.array([[5., 0., .5], [5., .2, .5]])]
    for i, points in enumerate(regions):
        snapshot = source.latest()
        snapshot.observation['position'] = np.array([i*4., 0., .3])
        snapshot.observation['sensors']['lidar']['position'] = np.array([i*4., 0., .4])
        snapshot.observation['sensors']['lidar']['pointcloud'] = points
        tasks.update(snapshot, snapshot.captured_at)
        session.last_publish = float('-inf')
        session.update(snapshot, tasks, 'disabled')
        wait_until(lambda: session.frame_id == i+1)
    displayed = np.asarray(session.display()['cloud']['points'])
    for point in np.concatenate(regions):
        assert np.min(np.linalg.norm(displayed-point, axis=1)) < .01
    assert session.display()['cloud']['count'] == 4
    # A new view capture and switching tasks both keep the same accumulated map.
    tasks.start('navigate', (6., 0., .3), snapshot.captured_at)
    np.testing.assert_allclose(SceneCapture().capture(snapshot, tasks)['points'], displayed)


def test_cloud_export_fits_old_and_new_distant_regions():
    from deploy.visualization.cloud_image import render_cloud
    points = np.array([[-100., 0., .5], [100., 0., .5]])
    image = render_cloud(points, [100., 0., .3])
    background = image[0, 0]
    rows, cols = np.where(np.any(image != background, axis=2))
    assert cols.min() > 0 and cols.max() < 719
    assert cols.max()-cols.min() > 500  # both regions, not only robot surroundings
    assert rows.min() > 0 and rows.max() < 719
