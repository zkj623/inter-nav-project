"""Display voxel ownership and spawn protocol; no ROS or SDK imports."""
from dataclasses import asdict
import multiprocessing as mp
import time

import numpy as np
import pytest

from deploy.robot.observation import Snapshot
from deploy.visualization.worker import FrameRenderer, process_frames
from grutopia_extension.interactive_navigation.mapping import MappingConfig, VoxelPointCloudMap


def item(step=0, points=None, *, publish=True, ready=True, captured_at=None):
    points = np.asarray([[1.05, .05, .55]] if points is None else points, dtype=np.float32)
    stamp = float(step+1) if captured_at is None else captured_at
    origin = np.array([.05, .05, .55])
    snapshot = Snapshot({'position': np.array([0., 0., .3]),
                         'orientation': np.array([1., 0., 0., 0.]),
                         'sensors': {'lidar': {'pointcloud': points, 'position': origin,
                                               'physics_step': step, 'max_range': 8.}}},
                        stamp+.1, stamp+.05, stamp, None, (0., 0., 0.), step,
                        geometry_calibrated=ready, planning_frame='experiment')
    mask = np.zeros((20, 20), dtype=bool)
    scene = {'ready': ready, 'reason': 'unverified geometry', 'updated_at': stamp,
             'position': snapshot.observation['position'].copy(),
             'orientation': snapshot.observation['orientation'].copy(),
             'config': asdict(MappingConfig(x_limits=(-1., 1.), y_limits=(-1., 1.))),
             'observed': ~mask, 'occupied': mask, 'inflated': mask.copy(),
             'skeleton': [], 'trajectory': [], 'path': [], 'frontiers': [], 'objects': [],
             'goal': None, 'revision': step+1, 'frame': 'experiment'}
    return {'scan': {'pointcloud': points, 'position': origin, 'physics_step': step,
                     'max_range': 8., 'captured_at': stamp, 'step': step},
            'snapshot': snapshot if publish else None, 'scene': scene if publish else None,
            'queued_at': time.monotonic(), 'capture_ms': 1.}


def test_worker_keeps_every_scan_and_original_voxel_clear_semantics():
    config = {'voxel_clear_observations': 3}
    renderer = FrameRenderer(config, True)
    reference = VoxelPointCloudMap(MappingConfig(**config))
    old, unseen, farther = [1.05, .05, .55], [1.05, 1.05, .55], [3.05, .05, .55]
    for step, points in enumerate(([old, unseen], [farther], [farther], [farther])):
        packet = item(step, points, publish=step == 3)
        scan = packet['scan']
        reference.clear_lidar_free_space(scan['position'], scan['pointcloud'], scan['max_range'])
        reference.update(scan['pointcloud'], step, 'lidar')
        output = renderer.render(packet)
        if step < 3:
            assert output is None
    np.testing.assert_array_equal(output['scene']['points'], reference.points())
    assert not np.any(np.linalg.norm(output['scene']['points']-old, axis=1) < .01)
    assert np.any(np.linalg.norm(output['scene']['points']-unseen, axis=1) < .01)
    assert output['scene']['voxel_count'] == 2
    assert 'points' not in packet['scene']  # Output metadata is detached.


def test_repeated_scan_cannot_clear_geometry_or_refresh_cloud_timestamp():
    renderer = FrameRenderer({'voxel_clear_observations': 3}, True)
    old, farther = [1.05, .05, .55], [3.05, .05, .55]
    renderer.render(item(0, [old]))
    renderer.render(item(1, [farther], captured_at=2.))
    for _ in range(5):
        output = renderer.render(item(1, [farther], captured_at=99.))
    assert output['cloud']['stamp'] == 2.
    assert output['scene']['cloud_physics_step'] == 1
    assert np.any(np.linalg.norm(output['scene']['points']-old, axis=1) < .01)


def test_cloud_and_2d_map_keep_their_own_source_timestamps():
    renderer = FrameRenderer({}, True)
    renderer.render(item(0, captured_at=1., publish=False))
    packet = item(1, captured_at=2.)
    packet['scan'] = None  # Parent did not accept/fuse this scan.
    output = renderer.render(packet)
    assert output['cloud']['stamp'] == 1.
    assert output['scene']['cloud_updated_at'] == 1.
    assert output['scene']['updated_at'] == 2.
    assert output['pose']['stamp'] == 2.05
    assert output['snapshot'] is packet['snapshot']


def test_out_of_order_scan_does_not_reset_history():
    renderer = FrameRenderer({}, True)
    output = renderer.render(item(2))
    original = output['scene']['points'].copy()
    with pytest.raises(ValueError, match='out-of-order'):
        renderer.render(item(1, [[5., 0., .5]]))
    output = renderer.render(item(2))
    np.testing.assert_array_equal(output['scene']['points'], original)


def test_raw_unverified_cloud_does_not_claim_cumulative_map():
    packet = item(0, ready=False)
    packet['scan'] = None
    output = FrameRenderer({}, True).render(packet)
    assert not output['map']['ready'] and not output['cloud']['accumulated']
    assert output['cloud']['stamp'] == packet['snapshot'].lidar_at
    np.testing.assert_allclose(output['cloud']['points'], packet['snapshot'].observation['sensors']['lidar']['pointcloud'])


def test_unreceived_cloud_is_not_stamped_as_current():
    packet = item(0)
    packet['scan'] = None
    with pytest.raises(ValueError, match='not received'):
        FrameRenderer({}, True).render(packet)


def test_existing_main_process_voxels_and_rgbd_preview_remain_supported():
    packet = item(0)
    points = np.array([[2., 0., .6]], dtype=np.float32)
    packet['scene'].update(points=points, voxel_count=1)
    packet['scan'] = None
    packet['snapshot'].camera_at = .9
    packet['snapshot'].observation['sensors']['camera'] = {'rgba': np.zeros((8, 8, 4), dtype=np.uint8)}
    output = FrameRenderer({}, False).render(packet)
    np.testing.assert_array_equal(output['scene']['points'], points)
    assert output['cloud']['stamp'] == packet['scene']['updated_at']
    assert output['rgb']['stamp'] == .9 and output['rgb']['jpeg']
    assert output['timings']['capture_ms'] == 1.


def test_spawn_worker_reports_error_continues_and_drains_in_order():
    context = mp.get_context('spawn')
    incoming = context.Queue(4)
    outgoing, send = context.Pipe(duplex=False)
    process = context.Process(target=process_frames, args=(incoming, send, {}, True))
    process.start()
    send.close()
    def receive(timeout=5):
        assert outgoing.poll(timeout), "worker did not return a result"
        return outgoing.recv()
    try:
        assert receive(10) == {'ready': True}
        incoming.put(item(2, publish=False), timeout=2)
        incoming.put(item(1), timeout=2)
        error = receive()
        assert error['frame_dropped'] and error['cloud_failed']
        assert 'out-of-order' in error['error']
        incoming.put(item(3), timeout=2)
        incoming.put(None, timeout=2)
        output = receive()
        assert output['scene']['cloud_physics_step'] == 3
        assert receive() is None
        process.join(timeout=2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        incoming.cancel_join_thread()
        incoming.close()
        outgoing.close()
