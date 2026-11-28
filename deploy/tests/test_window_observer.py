"""Read-only window observations alongside the ordinary navigation provider."""
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import ROOT, load
from deploy.interaction.config import box_window_config, create_perception
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


def test_preserved_old_field_preset_uses_upper_end_and_respects_explicit_override(tmp_path):
    import yaml
    payload = yaml.safe_load((ROOT/'configs/box_30x80x30.yaml').read_text())
    del payload['first_push_face']
    payload['contact_profile'] = str(ROOT/'configs'/payload['contact_profile'])
    field = tmp_path/'box_30x80x30.yaml'
    field.write_text(yaml.safe_dump(payload))
    assert load(interaction_path=field)['interaction']['first_push_face'] == 'upper_short_end'
    custom = tmp_path/'other.yaml'
    custom.write_text(field.read_text())
    assert load(interaction_path=custom)['interaction']['first_push_face'] is None
    field.write_text(yaml.safe_dump(dict(payload, first_push_face=None)))
    assert load(interaction_path=field)['interaction']['first_push_face'] is None


def latest_fit(points, sources, *args):
    return [WindowFit(np.zeros(2), 0., int(sources[-1]), .005)]


def test_sparse_top_confirms_both_ends_of_a_partially_visible_box():
    from deploy.perception.registered_boxes import DEFAULT_TRACKING
    from deploy.perception.window_fitting import verify
    corner = np.array([.9, -.2])
    size, origin = np.array([.3, .8]), np.array([0., -.8, .46])
    def side(axis, low, high):
        distance, z = np.meshgrid(np.linspace(low, high, 16), np.linspace(.045, .265, 8))
        xyz = np.column_stack((distance.ravel(), np.zeros(z.size), z.ravel()))
        return xyz[:, [axis, 1-axis, 2]]
    long_side = side(1, .04, .78)
    short_side = side(0, .105, .275)
    x, y = np.meshgrid([.018, .065, .15, .235, .282], [.02, .16, .4, .64, .78])
    top = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .32)))
    def fit(parts, *, confirmed=False, repeated=False):
        points = np.concatenate(parts)+np.r_[corner, 0.]
        labels = np.repeat([0, 0 if repeated else 1], len(points))
        return verify(np.tile(points, (2, 1)), labels, origin, size, .30, corner, 0.,
                      DEFAULT_TRACKING, confirmed=confirmed)
    sides = [long_side, short_side]
    result = fit([*sides, top])
    assert result.surfaces == 'top_and_side' and result.support_index == 1
    np.testing.assert_allclose(result.center, corner+size/2, atol=.005)
    assert fit(sides) is None
    assert fit([*sides, top[top[:, 0] > .06]]) is None
    assert fit([*sides, top, top+[0., 0., .12]]) is None
    assert fit([*sides, top], repeated=True) is None
    narrow_side = side(0, .15, .26)
    assert fit([long_side, narrow_side], confirmed=True).surfaces == 'two_sides'
    assert fit([long_side], confirmed=True) is None
    assert fit([side(1, .3, .55), narrow_side], confirmed=True) is None


def test_stationary_window_never_reuses_the_same_scans_for_two_votes(monkeypatch):
    windows = []
    def measure(points, sources, *args):
        windows.append((int(sources[0]), int(sources[-1])))
        return latest_fit(points, sources)
    fixed_fit(monkeypatch, measure)
    observer = WindowBoxObserver({**settings()[2], 'stationary_window_s': 2.})
    for i in range(20):
        assert not sample(observer, i)['valid']
    assert sample(observer, 20)['valid']
    assert len(windows) == 2 and windows[1][0] == 10
    # Scans older than a second come from the previous pose.
    packet = scan(21, 12.1)
    packet.observation['sensors']['lidar']['position'][0] += .12
    observer.update(packet, packet.captured_at)
    assert observer.last_delivery['window_start_at'] > 11.1


def test_wide_association_gate_checks_for_another_box_before_refreshing_partial_sides(monkeypatch):
    from deploy.perception import window_fitting as fitting
    from deploy.perception.registered_boxes import DEFAULT_TRACKING
    known = WindowFit(np.array([1., 0.]), 0., 0, .004, 0.)
    measured = WindowFit(known.center.copy(), 0., 4, .005, 0.)
    neighbor = WindowFit(known.center+[0., .35], 0., 4, .005, 0.)
    alternatives = []
    monkeypatch.setattr(fitting, 'verify', lambda *a, **kw: measured if kw['confirmed'] else None)
    monkeypatch.setattr(fitting, 'find_window_boxes', lambda *a: alternatives)
    args = (np.empty((0, 3)), np.empty(0), np.array([0., 0., .46]), (.8, .3, .3),
            DEFAULT_TRACKING, known, .40, .1)
    assert fitting.refit_window_box(*args) == []
    assert fitting.refit_window_box(*args, confirmed=True) == [measured]
    alternatives.append(neighbor)
    assert fitting.refit_window_box(*args, confirmed=True) == [neighbor, measured]


def test_native_rectangle_with_only_one_side_requires_a_bounded_top_from_distinct_scans():
    from deploy.perception.registered_boxes import DEFAULT_TRACKING
    from deploy.perception.window_fitting import find_window_boxes, refit_window_box
    angle = .42
    axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([1.2, -.4])
    u, v = np.meshgrid(np.linspace(-.40, .40, 25), np.linspace(-.15, .15, 13))
    top = np.c_[u.ravel(), v.ravel(), np.full(u.size, .312)]
    a, z = np.meshgrid(np.linspace(-.39, .39, 25), np.linspace(.09, .27, 9))
    side = np.c_[a.ravel(), np.full(a.size, -.15), z.ravel()]
    origin = np.r_[center + axes @ [0., -1.], .45]
    dims, region = (.3, .8, .3), (.4, 2., -.9, .4)
    def data(parts):
        points = np.concatenate(parts).copy()
        points[:, :2] = points[:, :2] @ axes.T+center
        return np.tile(points, (2, 1)), np.repeat([0, 1], len(points))
    points, ids = data([side, top])
    args = (origin, dims, region, DEFAULT_TRACKING)
    fit, = find_window_boxes(points, ids, *args)
    assert fit.surfaces == 'top_and_side' and fit.support_index == 1
    np.testing.assert_allclose(fit.center, center, atol=.02)
    assert abs((fit.long_axis_rad-angle+np.pi/2) % np.pi-np.pi/2) < .04
    assert refit_window_box(points, ids, origin, dims, DEFAULT_TRACKING, fit, .05, .1, search_region=False)
    assert not find_window_boxes(points, ids*0, *args)
    assert not find_window_boxes(points, ids, np.r_[origin[:2], .25], *args[1:])
    for parts in ([top], [side], [side, top[abs(top[:, 1]) < .04]],
                  [side, top, top+[0., 0., .12]], [side, top, top+[.7, 0., 0.]]):
        assert not find_window_boxes(*data(parts), *args)
    field_args = (*args[:-1], dict(DEFAULT_TRACKING, use_box_top=True))
    points, ids = data([top])
    lid, = find_window_boxes(points, ids, *field_args)
    assert lid.surfaces == 'top' and lid.support_index == 1
    np.testing.assert_allclose(lid.center, center, atol=.02)
    assert not find_window_boxes(points, ids*0, *field_args)
    for parts in ([top[abs(top[:, 1]) < .04]], [top, top+[0., 0., .12]], [top, top+[.7, 0., 0.]]):
        assert not find_window_boxes(*data(parts), *field_args)


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
    observer = WindowBoxObserver(box_window_config(config))
    for index in range(21):
        packet = scan(index, 10+index/10)
        points = packet.observation['sensors']['lidar']['pointcloud']
        points *= [.30/.40, .80/.30, .30/.50]  # scale both visible faces of the fixture
        if swap_xy:
            points[:, :2] = points[:, [1, 0]]
        points[:, 0] += 1.
        field_scan = packet.navigation_view(config['navigation']['mapping']['point_height'])
        assert observer.update(field_scan, field_scan.captured_at) == ((), None)
    row = observer.last_delivery['detections'][0]
    np.testing.assert_allclose(row['center'], (1., 0., .15), atol=.02)
    error = (row['long_axis_rad']-expected_axis+np.pi/2) % np.pi-np.pi/2
    assert row['valid'] and abs(error) < np.deg2rad(5) and not row['push_allowed']
    assert row['surfaces'] == 'two_sides'


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
        box_window_config(config)
    config['interaction'].update(settings()[2], tracking_enabled=True)
    assert box_window_config(config)['boxes'] == config['interaction']['boxes']
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
    from deploy.interaction.config import box_window_height_limits
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    window = WindowBoxObserver(box_window_config(cfg))
    single = create_perception(cfg, synchronous=True)
    try:
        assert window.config['max_age_s'] == .35
        assert single.config['max_age_s'] == .38
        assert box_window_height_limits(cfg) == pytest.approx([-.1, .46])
        assert cfg['navigation']['interaction_enabled'] and cfg['interaction']['first_push']
        assert cfg['interaction']['contact_profile']['calibration_reference'] is None
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


def test_box_process_uses_the_navigation_scan_stamp_and_reports_a_sensor_gap():
    from queue import Queue
    import time
    from deploy.perception.window_process import WindowProcess

    results = Queue()
    sink = SimpleNamespace(event=lambda name, **result: results.put((name, result)))
    worker = WindowProcess.from_navigation_scans(settings()[2], sink, 'box_window_result', None)
    try:
        stamp = time.monotonic()
        worker.submit(scan(52, stamp))
        name, result = results.get(timeout=3.)
        assert name == 'box_window_result' and result['scan_id'] == 52
        assert result['captured_at'] == stamp
        worker.submit(None, 'scan unavailable')
        _, invalid = results.get(timeout=3.)
        assert invalid['error'] == 'scan unavailable' and not invalid['detections']
    finally:
        worker.close()


def test_full_box_queue_keeps_the_invalidation_for_the_next_measured_scan():
    from queue import Queue
    from threading import Event
    from deploy.perception.window_process import WindowProcess

    worker = WindowProcess.__new__(WindowProcess)
    worker.scans, worker.stop, worker.pending_scan_error = Queue(maxsize=1), Event(), None
    first, latest = scan(1, 10.), scan(3, 10.2)
    worker.submit(first)
    worker.submit(None, 'invalid scan')
    worker.submit(scan(2, 10.1))
    assert worker.scans.get_nowait() == (None, first)
    latest.observation['sensors']['camera'] = {'rgba': np.zeros((8, 8, 3))}
    worker.submit(latest)
    error, queued = worker.scans.get_nowait()
    assert error == 'invalid scan' and queued.lidar_at == latest.lidar_at
    assert set(queued.observation['sensors']) == {'lidar'}
    assert queued.observation['sensors']['lidar'] is latest.observation['sensors']['lidar']
    assert 'camera' in latest.observation['sensors']
    assert worker.pending_scan_error is None


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
