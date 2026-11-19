"""Observation-only routing, independent window votes and source-timed tracking."""
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


def fixed_fit(monkeypatch):
    def detect(points, sources, origin, dimensions, region, config):
        return [WindowFit(np.zeros(2), 0., int(sources[-1]), .005)]
    monkeypatch.setattr(module, 'find_window_boxes', detect)


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


def test_missing_expiry_and_reacquisition(monkeypatch):
    fixed_fit(monkeypatch)
    observer = WindowBoxObserver(settings()[2])
    for i in range(21):
        before = sample(observer, i)
    fit = module.find_window_boxes
    monkeypatch.setattr(module, 'find_window_boxes', lambda *args: [])
    for i in range(21, 30):
        row = sample(observer, i)
        assert not row['valid'] and row['track_id'] == before['track_id']
        assert 'center' not in row and 'expires_at' not in row
    assert sample(observer, 30)['track_id'] is None
    monkeypatch.setattr(module, 'find_window_boxes', fit)
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
    monkeypatch.setattr(module, 'find_window_boxes', lambda *a: [WindowFit(np.zeros(2), 0., 0, .005)])
    observer = WindowBoxObserver(settings()[2])
    for i in range(21):
        row = sample(observer, i)
    assert not row['valid'] and row['invalid_reason'] == 'surface_evidence_expired'


def test_slow_live_fit_expires_before_publication(monkeypatch):
    fixed_fit(monkeypatch)
    ticks=iter(j/2 for j in range(50))
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    observer = WindowBoxObserver(settings()[2], live=True)
    for i in range(21):
        row = sample(observer, i)
    assert not row['valid'] and row['invalid_reason'] == 'surface_evidence_expired'


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
    monkeypatch.setattr(module, 'find_window_boxes', old_face)
    row = sample(observer, 21)
    assert row['tracking_state'] == 'missing' and 'center' not in row
    assert row['invalid_reason'] == 'surface_evidence_expired'


def test_factory_and_cli_isolate_window_observation():
    config = load()
    with pytest.raises(ValueError, match='tracking_enabled'):
        create_perception(config, box_window=True)
    config['interaction'].update(settings()[2], tracking_enabled=True)
    assert isinstance(create_perception(config, box_window=True), WindowBoxObserver)
    single = create_perception(config, synchronous=True)
    assert type(single) is RegisteredBoxPerception
    single.close()
    for flags in ([], ['--allow-motion'], ['--demo'], ['--motion-test'],
                  ['--replay', 'unused', '--task', 'navigate', '--goal', '1', '0']):
        with pytest.raises(SystemExit) as e:
            main(['--box-window', *flags])
        assert e.value.code == 2
