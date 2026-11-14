"""Actual callback calls, raw obstacle ownership and unknown-space preservation."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from deploy.app.navigation import NavigationEngine
from deploy.tests.test_online_navigation import config
from deploy.tests.test_interaction_execution import profile
from deploy.tests.test_registered_boxes import cloud
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


def fixture():
    p = profile()
    engine = NavigationEngine(config(), interaction={'contact_profile': p.to_dict()})
    checker = engine.contact_collision
    # Synthetic independently observed empty room, then measured target returns.
    checker.evidence.observed[:] = True
    checker.evidence.log_odds[:] = -4.
    dims = (.4, .3, .5)
    box = BoxState('B1', (1., .0), dims, local_dimensions=dims,
                   orientation=(1., 0., 0., 0.), center_z=.25)
    checker.points = cloud(center=box.xy, yaw=0., noise=0.)
    checker.last_scan = 1
    ctx = SimpleNamespace(box=box, boxes=[box], position=np.array([.1, 0., .3]),
                          heading=0., body_velocity=np.zeros(3))
    return engine, checker, ctx


def test_checked_clear_contact_path_is_actually_executable_not_typeerror():
    _, checker, ctx = fixture()
    before = checker.evidence.log_odds.copy()
    assert checker([.05, 0., 0.], ctx), checker.diagnostic
    assert checker.diagnostic['reason'] == 'measured_contact_sweep_clear'
    np.testing.assert_array_equal(before, checker.evidence.log_odds)


def test_non_target_return_is_not_hidden_by_whole_box_aabb():
    _, checker, ctx = fixture()
    checker.points = np.vstack([checker.points, [.63, 0., .3]])
    assert not checker([.15, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'non_target_body_sweep_blocked'


def test_unknown_box_destination_is_not_cleared_by_target_motion():
    _, checker, ctx = fixture()
    # Cell ahead of the box (not behind the robot) has no free-ray evidence.
    region = checker.xy[..., 0] > 1.22
    checker.evidence.observed[region] = False
    before = checker.evidence.observed.copy()
    assert not checker([.20, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_blocked_or_unknown'
    np.testing.assert_array_equal(before, checker.evidence.observed)


def test_missing_current_target_surface_blocks_contact():
    _, checker, ctx = fixture()
    checker.points = np.array([[2., -1., .4], [2., -1.1, .4], [2., -1.2, .4]])
    assert not checker([.05, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'target_surface_not_measured'


def test_box_corners_must_remain_inside_map_even_if_center_is_inside():
    _, checker, ctx = fixture()
    ctx.box = replace(ctx.box, xy=(2.85, 0.))
    ctx.boxes = [ctx.box]
    checker.points = cloud(center=ctx.box.xy, yaw=0., noise=0.)
    assert not checker([.05, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_outside_map'


def test_retreat_uses_translation_stopping_response(monkeypatch):
    from deploy.interaction import collision
    _, checker, ctx = fixture()
    checker.profile = replace(checker.profile, translation_stop_seconds=1.5, push_stop_seconds=.2)
    horizons = []
    monkeypatch.setattr(collision, 'body_motion_clear', lambda *args, **kw: horizons.append(kw['horizon']) or True)
    ctx.body_velocity = np.array([.1, 0., 0.])
    assert checker([-.05, 0., 0.], ctx, pushing=False)
    retreat = horizons[-1]
    assert checker([.05, 0., 0.], ctx, pushing=True)
    assert retreat > horizons[-1]


def test_measured_lateral_drift_is_checked_against_raw_non_target_returns():
    _, checker, ctx = fixture()
    ctx.body_velocity = np.array([0., .2, 0.])
    checker.points = np.vstack([checker.points, [.3, .42, .3]])
    assert not checker([.05, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'measured_body_stop_sweep_blocked'
