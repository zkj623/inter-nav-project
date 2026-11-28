"""Actual callback calls, raw obstacle ownership and unknown-space preservation."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from deploy.app.navigation import NavigationEngine
from deploy.tests.test_online_navigation import config
from deploy.tests.test_interaction_execution import profile
from deploy.tests.test_registered_boxes import cloud
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


def fixture(navigation=None):
    p = profile()
    engine = NavigationEngine(navigation or config(), interaction={'contact_profile': p.to_dict()})
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


@pytest.mark.parametrize('stop', ['timer', 'travel', 'gap_timeout'])
def test_aligned_field_push_times_the_stroke_from_odometric_face_arrival(stop):
    from deploy.app.config import ROOT, load
    from deploy.interaction.first_push import SHORT_END_PUSH_DISTANCE_M
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import permission

    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'])
    # Launch pose, face range, and time of the braking-grid veto in run 534084f4.
    box = BoxState('box-1', (1.295, -.825), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    plan = SimpleNamespace(box_id=box.id, direction=(0., -1.), push_distance=SHORT_END_PUSH_DISTANCE_M)
    engine.skill = create_push_skill(permission(object_id=box.id), plan, (0., 0.), (4., 0.),
                                     engine.contact_profile, upper_short_end=True)
    engine.skill.contact_started = engine.now = 10.
    engine.position = np.array([1.295, .278, .32])
    engine.heading = -np.pi/2
    engine.linear = engine.angular = np.zeros(3)
    engine.tilt = 0.
    engine.layers.decision = {}
    ctx = SimpleNamespace(box=box, position=engine.position, heading=engine.heading,
                          orientation=(np.sqrt(.5), 0., 0., -np.sqrt(.5)))
    engine.execution.start(box.id, engine.now)
    engine.observation = {'lidar_captured_at': engine.now}
    engine.planner.free[:] = engine.planner.box_free[:] = False
    engine.contact_collision.evidence.static_occupied[:] = True
    engine.contact_collision.points = np.array([[1.30, -.24, .17]])
    np.testing.assert_allclose(engine._start_timed_push(ctx, .38), [.35, 0., 0.])
    push = engine.skill.timed_push
    assert push.stroke_seconds == pytest.approx((.40/.35) * 4/3)
    assert not engine.trial_used
    engine.push_speed_sent(box.id, interaction_id=engine.interaction_command['id'], worker_at=10.03)

    def step(elapsed, forward):
        engine.now = push.sdk_started_at + elapsed
        engine.observation['lidar_captured_at'] = engine.now - .1
        # Lateral drift cannot count towards the initial gap.
        engine.position[:2] = push.origin + forward * push.forward + [.07, 0.]
        return engine._timed_push_step()

    np.testing.assert_allclose(step(1.3736, .18), [.35, 0., 0.])
    np.testing.assert_allclose(step(3.00, push.gap-.02), [.35, 0., 0.])
    assert push.face_reached_at is None
    if stop == 'gap_timeout':
        assert step(push.approach_timeout+.01, push.gap-.04) is None
        assert engine.result['reason'] == 'push_aside:timed_push_gap_not_reached'
        return
    np.testing.assert_allclose(step(3.10, push.gap+.005), [.50, 0., 0.])
    assert engine.interaction_command['forward_duration_s'] == pytest.approx(3.10 + push.stroke_seconds)
    # Passing both former stop thresholds still leaves a forward stroke.
    np.testing.assert_allclose(step(3.10+.40/.35+.01, push.gap+.41), [.50, 0., 0.])
    elapsed = 3.10 + push.stroke_seconds
    if stop == 'travel':
        np.testing.assert_allclose(step(elapsed-.008, push.gap+push.distance-.02), [.20, 0., 0.])
        np.testing.assert_allclose(step(elapsed-.005, push.gap+push.distance), [0., 0., 0.])
    else:
        np.testing.assert_allclose(step(elapsed+.01, push.gap+.45), [0., 0., 0.])
    assert engine.phase == 'push_settle'
    done, = [event for event in engine.events if event['event'] == 'timed_push_elapsed']
    assert done['reason'] == ('forward_travel' if stop == 'travel' else 'stroke_timer')



@pytest.mark.parametrize('same_cell_obstacle', [False, True])
@pytest.mark.parametrize('window_only', [False, True])
def test_box_edge_cell_belongs_to_the_push_target_only_with_exclusive_returns(same_cell_obstacle, window_only):
    from deploy.app.config import ROOT, load
    from deploy.interaction.first_push import trial_sweep_clear
    from deploy.perception.box_memory import BoxMapMemory
    from deploy.tests.test_online_navigation import observation
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    memory = BoxMapMemory()
    engine = NavigationEngine(config(), field, box_memory=memory)
    layers, planner = engine.layers, engine.planner
    box = BoxState('box-1', (1.02, .11), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    points = [[.87, .01, z] for z in (.13, .17, .21)]
    cell = planner.cell(points[0])
    assert not planner.box_mask(box)[cell]  # Centre is outside; measured face crosses this cell.
    if same_cell_obstacle:
        points.append([.815, .01, .17])
        assert planner.cell(points[-1]) == cell
    layers.observed[:] = True
    layers.evidence.log_odds[:] = -4.
    layers.evidence.log_odds[cell] = 3.  # Box edge previously mapped without identity.
    layers.local_hits[cell] = True
    if window_only:
        row = dict(object_id=box.id, track_id='0:box-1:1', valid=True, captured_at=.95, expires_at=1.15,
                   center=[*box.xy, box.center_z], dimensions_m=box.local_dimensions, long_axis_rad=np.pi/2)
        memory.observe(dict(geometry_source='window_two_faces', detections=[row]), 1.)
    engine.observe(observation(1, points=points), 1., [] if window_only else [box])
    assert engine._fresh_object(box.id) == (not window_only)
    assert bool(layers.tracked_occupied[cell]) == (not same_cell_obstacle)
    assert bool(layers.local_occupied[cell])  # The target remains a navigation obstacle.
    clear = trial_sweep_clear(planner, box, [1., 0.], [box], planner.free & planner.box_free, engine.contact_profile)
    assert clear == (not same_cell_obstacle)
    engine.observe(observation(2, points=points), 1.2, [])
    assert not planner.measured_box_cells and not planner.mixed_box_cells


def test_unknown_box_destination_is_not_cleared_by_target_motion():
    _, checker, ctx = fixture()
    # Cell ahead of the box (not behind the robot) has no free-ray evidence.
    region = checker.xy[..., 0] > 1.22
    checker.evidence.observed[region] = False
    before = checker.evidence.observed.copy()
    assert not checker([.20, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_blocked_or_unknown'
    np.testing.assert_array_equal(before, checker.evidence.observed)


def test_field_map_holds_lid_jitter_after_first_command_and_tracks_cumulative_motion():
    from deploy.app.config import ROOT, load
    from deploy.tests.test_online_navigation import observation

    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    engine = NavigationEngine(config(), field)
    engine.push_speed_sent('box-1')
    box = BoxState('box-1', (1., .052), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    recognition = SimpleNamespace(top_returns=lambda scan, now: {box.id: np.array([[1., 0., .3]])})
    for i, offset in enumerate((0., .015, -.020, .025, .04, .06), start=1):
        measured = replace(box, xy=(1., box.xy[1]-offset))
        engine.observe(observation(i, points=[[1.8, 1.2, .2]]), i*.2, [measured], recognition)
        assert engine.current_proposals[box.id].xy == measured.xy
        if i == 1:
            original = engine.layers.tracked_occupied.copy()
        if i == 2:
            assert (original ^ engine.planner.box_mask(measured)).any()
        if abs(offset) <= .05:
            assert engine.layers.detected_boxes[box.id].xy == box.xy
            np.testing.assert_array_equal(engine.layers.tracked_occupied, original)
        else:
            assert engine.layers.detected_boxes[box.id].xy == measured.xy
            assert (original ^ engine.layers.tracked_occupied).any()


def test_confirmed_field_box_releases_its_old_floor_and_ring_for_navigation():
    from deploy.app.config import ROOT, load
    from deploy.tests.test_online_navigation import observation

    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    engine = NavigationEngine(config(), field)
    layers, checker, planner = engine.layers, engine.contact_collision, engine.planner
    box = BoxState('box-1', (1.02, .11), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    edge = planner.cell((1.05, .55))
    old = planner.cell(box.xy)
    far = planner.cell((1.05, .85))
    for grid in (layers.evidence, checker.evidence):
        grid.log_odds[edge] = 4.
    layers.static_occupied[edge] = layers.local_hits[edge] = True

    def observe(i, measured, points):
        tops = {} if measured is None else {box.id: np.array([[measured.xy[0]+dx, measured.xy[1], .30]
                                                               for dx in (-.02, 0., .02)])}
        recognition = SimpleNamespace(top_returns=lambda scan, now: tops,
                                      snapshot=lambda now: {box.id: {'latest_at': now}})
        engine.observe(observation(i, points=points), i*.12, [measured] if measured else [], recognition)

    background = [[-.8, -1., .15]]
    observe(1, box, background)
    assert layers.observed[edge] and checker.evidence.observed[edge]
    assert planner.free[edge] and not layers.local_occupied[edge] and not checker.evidence.occupied_mask()[edge]
    assert not layers.observed[far]
    assert planner.blocked(layers.current_boxes)[old] and layers.global_occupied[old]
    engine.push_speed_sent(box.id)
    observe(2, None, background)
    assert layers.global_occupied[old]
    moved = replace(box, xy=(1.02, -.99))
    observe(3, moved, background)
    new = planner.cell(moved.xy)
    assert layers.global_occupied[new] and layers.local_occupied[new]
    for cell in (old, edge):
        assert layers.global_observed[cell] and layers.observed[cell] and checker.evidence.observed[cell]
        assert not layers.global_occupied[cell] and not layers.local_occupied[cell]
        assert planner.free[cell] and not checker.evidence.occupied_mask()[cell]
    # Complete the surrounding room survey so clearance tests the old box alone.
    layers.observed[:] = True
    layers.refresh_global()
    with layers.global_planning() as boxes:
        assert planner.path((.25, .15), (1.85, .15), boxes)
        assert not planner.blocked(boxes)[old] and planner.blocked(boxes)[new]
    observe(4, None, [[1.05, .15, .15]])
    assert layers.local_occupied[old]  # New measurements can still change the old floor.


def test_missing_current_target_surface_blocks_contact():
    _, checker, ctx = fixture()
    checker.points = np.array([[2., -1., .4], [2., -1.1, .4], [2., -1.2, .4]])
    assert not checker([.05, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'target_surface_not_measured'


def test_post_push_reset_preserves_obstacles_and_requires_rays_to_clear_the_old_box():
    from deploy.app.config import ROOT, load
    from deploy.tests.test_online_navigation import observation
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    engine = NavigationEngine(config(), field)
    layers, planner = engine.layers, engine.planner
    box = BoxState('box-1', (1., .10), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    clear = planner.cell((1.05, .15))
    mixed = planner.cell((1.05, .35))
    ring = planner.cell((1.05, .55))
    mask = np.zeros_like(layers.observed)
    mask[mixed] = True
    engine.observe(observation(1, points=[[.87, .01, z] for z in (.13, .17, .21)]), 1., [box])
    assert layers.tracked_occupied[clear] and layers.observed[ring]
    planner.mixed_box_cells[box.id] = mask
    engine.map_observed_at = .95
    engine._retire_object_tracks({box.id}, 'post_push_box_unknown', after_push=True)
    assert not layers.current_boxes and not layers.global_boxes
    assert not layers.observed[clear] and not planner.free[clear]
    assert layers.local_occupied[mixed] and layers.observed[ring]
    # A fresh pass on the exposed floor can establish it as ordinary walking space.
    packet = observation(2, points=[[2.1, .15, .02]])
    packet['sensors']['lidar']['position'] = np.array([0., .15, .3])
    engine.observe(packet, 1.2)
    assert layers.observed[clear] and planner.free[clear]


def test_trailing_short_edge_waits_for_a_new_scan_while_an_obstacle_ahead_blocks():
    from deploy.app.config import ROOT, load
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import permission

    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    engine = NavigationEngine(config(), field)
    checker = engine.contact_collision
    checker.evidence.observed[:] = True
    dims = (.8, .30, .30)
    box = BoxState('B1', (1., 0.), dims, local_dimensions=dims, orientation=(1., 0., 0., 0.), center_z=.15)
    checker.top_returns = {box.id: np.array([[1., y, .30] for y in (-.05, 0., .05)])}
    plan = SimpleNamespace(box_id=box.id, direction=(1., 0.), push_distance=.1)
    skill = create_push_skill(permission(), plan, (0., 0.), (2., 0.), engine.contact_profile, upper_short_end=True)
    ctx = SimpleNamespace(box=box, boxes=[box], position=np.array([-.35, 0., .32]), heading=0.,
                          body_velocity=np.zeros(3), layers=engine.layers, now=1.)
    skill.state, skill.contact_started = 'engage', .9
    cmd = [.12, 0., 0.]
    checker.points = np.array([[.54, .08, .15]])
    assert not checker(cmd, ctx)
    assert checker.diagnostic['reason'] == 'box_edge_needs_new_scan'
    wait = skill._fail(ctx, 'contact_motion_blocked')
    assert wait.status == 'running' and not any(wait.velocity) and skill.state == 'engage'
    checker.points = np.vstack([checker.points, [1.46, .08, .15]])
    assert not checker(cmd, ctx) and checker.diagnostic['reason'] == 'box_sweep_obstacle'
    checker.points = np.array([[2.5, .8, .15]])
    assert checker(cmd, ctx)


def test_throttled_map_keeps_contact_rays_and_lid_on_the_same_scan():
    from deploy.tests.test_online_navigation import observation

    engine, checker, ctx = fixture()
    layers, box = engine.layers, ctx.box
    first = observation(10, points=[[2., -1., .2]])
    lid = np.array([[1., y, .5] for y in (-.05, 0., .05)])
    layers.top_returns = {box.id: lid}
    layers.update(1., first, (box,), force=True)
    checker.update(first, layers)
    assert checker.target_measured(box, 10)
    next_scan = observation(11, points=[[2., 1., .2]])
    layers.top_returns = {}
    assert not layers.update(1.0996, next_scan, (box,))
    checker.update(next_scan, layers)
    assert checker.target_measured(box, 10) and not checker.target_measured(box, 11)
    last_global = layers.last_global
    assert layers.update(1.0996, next_scan, (box,), force_local=True)
    assert layers.last_global == last_global
    checker.update(next_scan, layers)
    assert checker.last_scan == 11
    assert not checker.target_measured(box, 10) and not checker.target_measured(box, 11)
    # A later delivery of the matching lid may fill the gap without fusing its rays twice.
    before = checker.evidence.log_odds.copy()
    layers.top_returns = {box.id: lid}
    checker.update(next_scan, layers)
    assert checker.target_measured(box, 11)
    np.testing.assert_array_equal(checker.evidence.log_odds, before)


def test_box_corners_must_remain_inside_map_even_if_center_is_inside():
    _, checker, ctx = fixture()
    ctx.box = replace(ctx.box, xy=(2.85, 0.))
    ctx.boxes = [ctx.box]
    checker.points = cloud(center=ctx.box.xy, yaw=0., noise=0.)
    assert not checker([.05, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_outside_map'


def test_box_cannot_be_pushed_over_field_wall_even_with_measured_free_cells():
    nav = config(world_boundary=True)
    nav['mapping']['x_limits'] = [-3., 1.24]
    _, checker, ctx = fixture(nav)
    assert checker.world_boundary.contains(ctx.box.xy)
    assert not checker([.15, 0., 0.], ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_outside_world_boundary'


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


@pytest.mark.parametrize('obstacle', ['return', 'contact_map', 'navigation_map', 'mixed', 'other_box'])
def test_field_box_can_slide_into_unknown_but_stops_for_obstacles(obstacle):
    from deploy.app.config import ROOT, load
    from deploy.interaction.first_push import trial_sweep_clear
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import permission
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    engine = NavigationEngine(config(), field)
    checker = engine.contact_collision
    checker.evidence.observed[:] = True
    checker.evidence.log_odds[:] = -4.
    box = BoxState('B1', (1., 0.), (.4, .3, .5), local_dimensions=(.4, .3, .5),
                   orientation=(1, 0, 0, 0), center_z=.25)
    checker.points = cloud(center=box.xy, yaw=0., noise=0.)
    # Unobserved space just beyond the far end of the box.
    checker.evidence.observed[checker.xy[..., 0] > 1.22] = False
    checker.last_scan = 1
    plan = SimpleNamespace(box_id=box.id, direction=(1., 0.), push_distance=.2)
    skill = engine.skill = create_push_skill(permission(), plan, (0., 0.), (2., 0.),
                                             engine.contact_profile, upper_short_end=True)
    skill.state = 'engage'
    ctx = SimpleNamespace(box=box, boxes=[box], position=np.array([-.1, 0., .32]),
                          heading=0., body_velocity=np.zeros(3), limiter=engine.limiter)
    command = np.array([.12, 0., 0.])
    assert engine._check_contact_motion(command, ctx)
    points = checker.points
    checker.points = np.vstack([points, [.45, 0., .2]])
    assert not engine._check_contact_motion(command, ctx)
    assert checker.diagnostic['reason'] == 'non_target_body_sweep_blocked'
    checker.points = points
    ctx.position[0] = .23
    unseen = checker.xy[..., 0] > 1.22
    layers, planner = engine.layers, engine.planner
    ctx.planner = planner
    layers.observed[:] = layers.global_observed[:] = ~unseen
    layers._bind_local()
    before = layers.observed.copy()
    assert skill.box_within_stopping_reach(ctx, command)
    assert engine._check_contact_motion(command, ctx)
    assert trial_sweep_clear(planner, box, plan.direction, ctx.boxes, planner.free & planner.box_free, engine.contact_profile)
    assert skill.backend.motion_clear(skill, ctx, command, contact=True)
    np.testing.assert_array_equal(layers.observed, before)
    assert not planner.free[unseen].any() and not checker.evidence.observed[unseen].any()
    # A single return in the otherwise unobserved box path must still win.
    xy = (1.25, .05)
    cell = planner.cell(xy)
    if obstacle == 'return':
        checker.points = np.vstack([points, [*xy, .17]])
    elif obstacle == 'contact_map':
        checker.evidence.log_odds[cell] = 4.
        checker.evidence.observed[cell] = True
    elif obstacle == 'navigation_map':
        layers.static_occupied[cell] = True
    elif obstacle == 'mixed':
        mixed = np.zeros_like(before)
        mixed[cell] = True
        planner.mixed_box_cells = {box.id: mixed}
    else:
        ctx.boxes.append(BoxState('neighbor', xy, (.12, .12, .24)))
    if obstacle in ('navigation_map', 'mixed', 'other_box'):
        assert not trial_sweep_clear(planner, box, plan.direction, ctx.boxes, planner.free & planner.box_free, engine.contact_profile)
        assert not skill.backend.motion_clear(skill, ctx, command, contact=True)
    assert not engine._check_contact_motion(command, ctx)
    assert checker.diagnostic['reason'] == 'box_sweep_obstacle'
    assert tuple(checker.diagnostic['cell_rc']) == planner.cell(checker.diagnostic['cell_xy_m'])
