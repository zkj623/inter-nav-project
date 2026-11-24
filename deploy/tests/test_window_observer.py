"""Read-only window observations alongside the ordinary navigation provider."""
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import load
from deploy.interaction.config import create_perception
from deploy.perception import window_observer as module
from deploy.perception.registered_boxes import RegisteredBoxPerception
from deploy.perception.window_fitting import WindowFit
from deploy.perception.window_observer import WindowBoxObserver
from deploy.run import main
from deploy.tests.test_registered_interaction_integration import scan, settings


def sample(observer, i, **kw):
    packet = scan(i, 10 + i/10, **kw)
    assert observer.update(packet, packet.captured_at) == ((), None)
    return observer.last_delivery['detections'][0]


def latest_fit(points, sources, *args):
    return [WindowFit(np.zeros(2), 0., int(sources[-1]), .005)]


def fixed_fit(monkeypatch, detect=latest_fit):
    monkeypatch.setattr(module, 'find_window_boxes', detect)
    monkeypatch.setattr(module, 'refit_window_box', lambda *args, **kwargs: detect(*args))


def test_actual_rectangular_box_requires_two_disjoint_windows_and_never_grants_motion():
    observer = WindowBoxObserver(settings()[2])  # even a verified, push-allowed registry
    for i in range(20):
        assert not sample(observer, i)['valid']
    row = sample(observer, 20)
    assert row['valid'] and not row['push_allowed']
    np.testing.assert_allclose(row['center'], (0., 0., .25), atol=.02)
    assert abs(row['face_angle_rad']) < .06 or np.pi/2-row['face_angle_rad'] < .06
    assert sample(observer, 21)['track_id'] == row['track_id']
    previous = observer.last_delivery
    assert observer.update(scan(21, 12.1), 12.12) == ((), None)
    assert observer.last_delivery is previous


def test_box_crossing_field_wall_cannot_acquire_a_track(monkeypatch):
    from deploy.robot.world_boundary import WorldBoundary
    fixed_fit(monkeypatch)
    observer = WindowBoxObserver(settings()[2], world_boundary=WorldBoundary((-2., 2.), (-3., .1)))
    for i in range(24):
        row = sample(observer, i)
        assert not row['valid'] and row['track_id'] is None
    assert row['invalid_reason'] == 'box_at_world_boundary'


@pytest.mark.parametrize('swap_xy,expected_axis', [(False, np.pi/2), (True, 0.)])
def test_long_low_box_reports_the_long_side_in_both_orientations(swap_xy, expected_axis):
    from deploy.app.config import ROOT
    config = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    observer = create_perception(config, box_window=True)
    for index in range(21):
        packet = scan(index, 10+index/10)
        points = packet.observation['sensors']['lidar']['pointcloud']
        points *= [.30/.40, .80/.30, .30/.50]  # scale both visible faces of the fixture
        if swap_xy:
            points[:, :2] = points[:, [1, 0]]
        points[:, 0] += 1.
        assert observer.update(packet, packet.captured_at) == ((), None)
    row = observer.last_delivery['detections'][0]
    np.testing.assert_allclose(row['center'], (1., 0., .15), atol=.02)
    error = (row['long_axis_rad']-expected_axis+np.pi/2) % np.pi-np.pi/2
    assert row['valid'] and abs(error) < np.deg2rad(5) and not row['push_allowed']


def test_missing_expiry_and_reacquisition(monkeypatch):
    fixed_fit(monkeypatch)
    observer = WindowBoxObserver(settings()[2])
    for i in range(21):
        before = sample(observer, i)
    fixed_fit(monkeypatch, lambda *args: [])
    for i in range(21, 30):
        row = sample(observer, i)
        assert not row['valid'] and row['track_id'] == before['track_id']
        assert 'center' not in row and 'expires_at' not in row
    assert sample(observer, 30)['track_id'] is None
    fixed_fit(monkeypatch)
    for i in range(31, 41):
        assert not sample(observer, i)['valid']
    assert sample(observer, 41)['track_id'] != before['track_id']
    assert sample(observer, 45)['invalid_reason'] == 'filling_window'


def test_duplicate_claims_and_old_surface_returns_are_invalid(monkeypatch):
    fixed_fit(monkeypatch)
    cfg = settings()[2]
    cfg['boxes'].append(dict(cfg['boxes'][0], id='B2'))
    observer = WindowBoxObserver(cfg)
    for i in range(22):
        sample(observer, i)
    assert all(r['invalid_reason'] == 'ambiguous' for r in observer.last_delivery['detections'])
    fixed_fit(monkeypatch, lambda *a: [WindowFit(np.zeros(2), 0., 0, .005)])
    observer = WindowBoxObserver(settings()[2])
    for i in range(21):
        row = sample(observer, i)
    assert not row['valid'] and row['invalid_reason'] == 'surface_evidence_expired'


def test_slow_stationary_discovery_only_provides_a_hint_until_a_fresh_local_fit(monkeypatch):
    fixed_fit(monkeypatch, lambda pts, ids, *args: [WindowFit(np.zeros(2), 0., int(ids[-1])-1, .005)])
    ticks = iter(j*.16 for j in range(50))
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    observer = WindowBoxObserver(settings()[2], live=True)
    for i in range(21):
        row = sample(observer, i)
    assert not row['valid'] and row['invalid_reason'] == 'await_fresh_refit'
    assert row['track_id'] is None and 'center' not in row
    # Scan 21 arrived during the second search. A fast fit on scan 22
    # still needs support from both faces after the candidate's last scan.
    observer.buffer.append(scan(21, 12.1), 12.1)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 0.))
    fixed_fit(monkeypatch, lambda pts, ids, *args: [WindowFit(np.zeros(2), 0., int(ids[-1])-2, .005)])
    row = sample(observer, 22)
    assert not row['valid'] and row['invalid_reason'] == 'await_fresh_refit'
    fixed_fit(monkeypatch)
    row = sample(observer, 23)
    assert row['valid'] and row['search'] == 'local'
    assert row['captured_at'] == pytest.approx(12.3)
    assert row['expires_at'] == pytest.approx(12.55) and not row['push_allowed']


def test_motion_blocks_first_confirmation_and_single_face_cannot_renew_a_track(monkeypatch):
    fixed_fit(monkeypatch)
    observer = WindowBoxObserver(settings()[2])
    for i in range(22):
        packet = scan(i, 10+i/10)
        packet.observation['sensors']['lidar']['position'][0] += i*.015
        observer.update(packet, packet.captured_at)
    assert observer.last_delivery['detections'][0]['invalid_reason'] == 'wait_for_stationary_window'
    observer.reset()
    for i in range(21):
        sample(observer, i)
    # Keep the earlier of the two face stamps fixed while a later scan of
    # the other face arrives. Window end time cannot renew the old face.
    def old_face(points, sources, *args):
        return [WindowFit(np.zeros(2), 0., int(sources[-1])-3, .005)]
    fixed_fit(monkeypatch, old_face)
    row = sample(observer, 21)
    assert row['tracking_state'] == 'missing' and 'center' not in row
    assert row['invalid_reason'] == 'surface_evidence_expired'


def test_factory_and_cli_route_the_window_without_changing_motion_mode(monkeypatch):
    from deploy.app import runtime
    config = load()
    with pytest.raises(ValueError, match='tracking_enabled'):
        create_perception(config, box_window=True)
    config['interaction'].update(settings()[2], tracking_enabled=True)
    assert isinstance(create_perception(config, box_window=True), WindowBoxObserver)
    single = create_perception(config, synchronous=True)
    assert type(single) is RegisteredBoxPerception
    single.close()
    for flags in (['--demo'], ['--motion-test'], ['--check'], ['--diagnose']):
        with pytest.raises(SystemExit) as e:
            main(['--box-window', *flags])
        assert e.value.code == 2
    for flags in (['--box-window'], ['--replay', 'unused']):
        with pytest.raises(SystemExit) as e:
            main(['--realtime', *flags])
        assert e.value.code == 2
    calls = []
    monkeypatch.setattr(runtime, 'run_live', lambda args, _: calls.append((args.box_window, args.allow_motion)))
    monkeypatch.setattr(runtime, 'run_replay', lambda args, _: calls.append((args.box_window, args.task)))
    for flags in ([], ['--observe-only'], ['--replay', 'unused', '--task', 'navigate', '--goal', '1', '0']):
        main(['--box-window', *flags])
    assert calls == [(True, True), (True, False), (True, 'navigate')]


def test_field_window_lease_is_separate_from_contact_evidence():
    from deploy.app.config import ROOT
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    window = create_perception(cfg, box_window=True)
    single = create_perception(cfg, synchronous=True)
    try:
        assert window.config['max_age_s'] == .35
        assert single.config['max_age_s'] == .25
        assert not cfg['navigation']['interaction_enabled']
    finally:
        window.close()
        single.close()


def test_replay_process_reports_a_lost_input_stream(tmp_path):
    import time
    from deploy.perception.window_process import WindowProcess
    errors = []
    sink = SimpleNamespace(event=lambda name, **data: errors.append(data))
    worker = WindowProcess.replay(settings()[2], sink, tmp_path, .28)
    worker.begin((10., time.monotonic()))
    with pytest.raises(RuntimeError, match='observations.jsonl'):
        worker.close(drain=True)
    assert errors[-1]['detections'] == []
    assert 'observations.jsonl' in errors[-1]['error']
    assert not worker.process.is_alive()


def test_moving_box_prefers_recent_faces_and_cannot_renew_a_fully_occluded_pose():
    observer = WindowBoxObserver(settings()[2])
    for i in range(24):
        initial = sample(observer, i)
    for i in range(24, 36):
        x = (i-23)*.015
        row = sample(observer, i, center=(x, 0.))
        assert row['valid'] and row['search'] in ('local_short', 'local_recent')
        assert row['track_id'] == initial['track_id']
        assert abs(row['center'][0]-x) < .045
        assert row['push_allowed'] is False
    for i in range(36, 40):
        lost = sample(observer, i, include_box=False)
    assert not lost['valid'] and 'center' not in lost and lost['track_id'] == initial['track_id']
