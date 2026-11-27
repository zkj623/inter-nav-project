"""Sensor/SDK boundary tests; shared algorithm tests live under tests/."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.app.config import load
from deploy.app.navigation import NavigationEngine
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


def config(**overrides):
    return {'mapping': {'x_limits': [-3., 3.], 'y_limits': [-3., 3.], 'grid_resolution': .1,
                        'robot_radius': .25, 'obstacle_height': [.05, .8]},
            'max_forward_speed': .30, 'max_lateral_speed': .15, 'max_yaw_speed': .4,
            'success_distance': .25, **overrides}


def room_navigation():
    # The synthetic ring and corridor span both sides of the robot.
    nav = load()['navigation']
    nav['mapping'].update(x_limits=[-5., 5.], y_limits=[-5., 5.])
    return nav


def observation(step=0, position=(0., 0., .4), velocity=(0., 0., 0.), points=None, full=True):
    if points is None:
        angle = np.linspace(-np.pi, np.pi, 1440, endpoint=False)
        points = np.column_stack((2.8*np.cos(angle), 2.8*np.sin(angle), np.full_like(angle, .4)))
    result = {'position': np.asarray(position), 'orientation': np.array([1., 0., 0., 0.]),
              'velocity': velocity,
              'sensors': {'lidar': {'position': np.array([0., 0., .4]), 'pointcloud': np.asarray(points, float),
                                    'physics_step': step, 'max_range': 5., 'min_range': .1}}}
    if full:
        result.update(body_linear_velocity=np.array([*velocity[:2], 0.]),
                      body_angular_velocity=np.array([0., 0., velocity[2]]))
    return result


BOX = BoxState('box', (1., 0.), (.4, .4, .5))
BOX_POINTS = [(0.8, -.05, .3), (.8, 0., .3), (.8, .05, .3)]


def test_engine_imports_without_isaac_and_starts_from_unknown_map():
    engine = NavigationEngine(config())
    assert not engine.layers.observed.any()
    assert not engine.layers.global_occupied.any()
    engine.set_goal((2.5, 0., .4), 0.)
    assert engine.status == 'running'
    assert not any(name.startswith(('omni.', 'isaacsim.')) for name in sys.modules)
    stats = engine.layers.statistics()
    assert stats['uses_simulator_object_poses'] is False
    assert 'simulated object proposals' not in stats['movable_pose_source']


def test_goal_contract_is_explicit():
    with pytest.raises(ValueError, match='success_distance'):
        NavigationEngine(config(success_distance=.30))


@pytest.mark.parametrize('goal', [(-3.01, 0.), (0., -3.01), (3., 0.), (0., 3.)])
def test_continuous_goal_bounds_do_not_truncate_negative_fractional_cells(goal):
    engine = NavigationEngine(config())
    with pytest.raises(ValueError, match='outside map bounds'):
        engine.set_goal(goal, 0.)


def test_real_range_boundary_return_cannot_reveal_space():
    engine = NavigationEngine(config())
    # The simulator treats a range shell as a miss. A ROS return at that
    # distance must not make the entire measured ray traversable.
    packet = observation(points=[(5., 0., .4)])
    engine.observe(packet, 0.)
    assert not engine.layers.observed[engine.planner.cell((2., 0.))]
    assert engine.layers.scan_count == 0
    np.testing.assert_array_equal(packet['sensors']['lidar']['pointcloud'], [(5., 0., .4)])


def test_startup_clear_area_is_fixed_once_and_an_actual_obstacle_wins():
    nav = load()['navigation']
    engine = NavigationEngine(nav)
    layer, cell = engine.layers, engine.planner.cell
    layer.capture_contact_evidence = True
    before = layer.observed.copy()
    untouched, edge = cell((.65, .35)), cell((.85, .35))
    obstacle = cell((.72, -.32))
    assert not before[untouched] and not before[edge]
    engine.observe(observation(points=[[.72, -.32, .18]]), .1)
    assert layer.observed[untouched] and layer.evidence.log_odds[untouched] == 0.
    assert not layer.measured_free[untouched] and not layer.observed[edge]
    assert layer.observed[obstacle] and layer.local_occupied[obstacle]
    initialized = [e for e in layer.events if e['event'] == 'startup_clear_area_initialized']
    assert len(initialized) == 1 and initialized[0]['cells'] > 0
    engine.set_goal((3., 0.), .2)
    engine.observe(observation(1, position=(1.5, 0., .4), points=[[2.4, -.3, .18]]), .5)
    assert not layer.observed[cell((2.15, .35))]
    assert sum(e['event'] == 'startup_clear_area_initialized' for e in layer.events) == 1
    nav['startup_clear_area'] = None
    other = NavigationEngine(nav)
    other.observe(observation(points=[[.72, -.32, .18]]), .1)
    assert not other.layers.observed[untouched]


def test_field_survey_sees_through_virtual_wall_but_exploration_stays_inside(tmp_path):
    from grutopia_extension.interactive_navigation.exploration_frontiers import local_unknown
    nav = load()['navigation']
    engine = NavigationEngine(nav)
    layers, cell = engine.layers, engine.planner.cell
    def packet(step, points, max_range=5.):
        obs = observation(step, points=points)
        obs['sensors']['lidar'].update(position=np.array([.05, .05, .18]), max_range=max_range)
        return obs
    # Only return is outside the north fence. Its ray crosses the field.
    engine.observe(packet(0, [[.05, 1.65, .18]]), 0.)
    inside = cell((.05, .75))
    assert layers.observed[inside] and not layers.local_occupied[inside]
    for n in range(1, 5):
        engine.observe(packet(n, [[.05, 2.15, .18]], max_range=8.), n*.4)
    field = layers.field_survey.snapshot(layers)
    def field_cell(x, y):
        return (int((y-field['y_limits'][0])/.1), int((x-field['x_limits'][0])/.1))
    assert field['observed'][field_cell(.05, 1.65)] and not field['occupied'][field_cell(.05, 1.65)]
    assert field['occupied'][field_cell(.05, 2.15)]
    assert not field['observed'][field_cell(.55, 1.65)]
    assert layers.observed[layers.boundary_wall].all() and layers.local_occupied[layers.boundary_wall].all()
    layers.refresh_global(reason='new_task')
    assert layers.global_occupied[layers.boundary_wall].all()
    with layers.global_planning():
        assert not engine.planner.free[~layers.interior].any()
        assert not engine.planner.path((0., 0.), (.05, 1.05), [])
    # A fully surveyed interior must never send Go2 to look over the fence.
    # Unknown cells in the real map at the wall stay unknown to the viewer.
    layers.global_observed[layers.interior & ~layers.boundary_goal_blocked] = True
    grid = layers.frontier_grid()
    assert not layers.global_observed[cell((1.25, .85))]
    assert grid.occupied_mask()[cell((1.25, .85))] and grid.observed.all()
    assert local_unknown(grid, (1.25, .65), radius=.8) == 0
    inner_gap = cell((1.25, .45))
    layers.global_observed[inner_gap] = False
    assert not layers.frontier_grid().observed[inner_gap]
    layers.save(tmp_path)
    saved = np.load(tmp_path/'field_map.npz')
    np.testing.assert_array_equal(saved['occupied'], field['occupied'])
    np.testing.assert_array_equal(saved['observed'], field['observed'])


def test_exterior_scan_without_any_interior_ray_cannot_start_motion():
    engine = NavigationEngine(load()['navigation'])
    engine.set_goal((4., 0.), 0.)
    obs = observation(points=[[-2., 0., .26]])
    obs['sensors']['lidar']['position'][2] = .55
    assert engine.update(obs, .1) is None
    assert engine.result['reason'] == 'no_lidar_returns_in_world_boundary'


def test_adjusted_field_bounds_and_unaligned_edges():
    nav = load()['navigation']
    nav['mapping'].update(x_limits=[-1.27, 4.96], y_limits=[-2.61, .93])
    engine = NavigationEngine(nav)
    inside = engine.layers.interior
    assert inside.sum() == 62*35
    assert not inside[engine.planner.cell((4.955, 0.))]
    for xy in ((4., 0.), (4., -1.4), (.2, .4)):
        engine.set_goal(xy, 0.)
        assert engine.goal == xy
    for xy in ((-.5, .93), (4.96, 0.), (-1.27, -1.), (1., -2.61), (.2, .65)):
        with pytest.raises(ValueError, match='实验边界'):
            engine.set_goal(xy, 0.)


@pytest.mark.parametrize('requested,measured', [((0., 0., .9), (0., 0., 0.)),
                                                ((0., -.10, 0.), (0., .28, 0.))])
def test_field_guard_checks_turning_corners_and_measured_stopping_motion(requested, measured):
    engine = NavigationEngine(load()['navigation'])
    engine.set_goal((2., 0.), 0.)
    engine.observe(observation(position=(1., .53, .3), velocity=measured, points=[[2., 0., .2]]), .1)
    assert engine._output(requested) is None
    np.testing.assert_array_equal(engine.limiter.command, np.zeros(3))
    assert engine.result['reason'] == 'world_boundary_stop'


def test_field_ceiling_rejects_high_rays_before_they_mark_free_cells_or_change_a_route():
    cfg = room_navigation()
    assert cfg['mapping']['point_height'] == [-.10, .28]
    assert cfg['mapping']['obstacle_height'] == [.10, .28]
    high = np.array([[2., 0., .28001], [1., 1., .30], [.7, 0., 1.5]])
    def packet(points):
        result = observation(points=points)
        result['sensors']['lidar']['position'][2] = .18
        return result

    engine = NavigationEngine(cfg)
    engine.observe(packet(high), 0.)
    assert engine.layers.scan_count == 0
    assert not engine.layers.observed[engine.planner.cell((1., 0.))]
    engine.observe(packet([[2., 0., float(np.float32(.28))]]), .2)
    assert engine.layers.evidence.log_odds[engine.planner.cell((2., 0.))] > 0
    engine.observe(packet([[1., .6, .09]]), .4)
    assert engine.layers.evidence.log_odds[engine.planner.cell((1., .6))] <= 0

    ring = observation()['sensors']['lidar']['pointcloud']
    ring[:, 2] = .23
    clean, mixed = NavigationEngine(cfg), NavigationEngine(cfg)
    for nav, points in ((clean, ring), (mixed, np.vstack((ring, high)))):
        sample = packet(points)
        nav.observe(sample, 0.)
        nav.set_goal((1., 0., .4), .1)
        nav.update(sample, .1)
    for layer in ('observed', 'global_observed', 'global_occupied', 'local_occupied'):
        np.testing.assert_array_equal(getattr(clean.layers, layer), getattr(mixed.layers, layer))
    np.testing.assert_array_equal(clean.layers.evidence.log_odds, mixed.layers.evidence.log_odds)
    assert clean.path and clean.decision['kind'] == mixed.decision['kind']
    np.testing.assert_allclose(clean.path, mixed.path)


def test_duplicate_scan_does_not_accumulate_evidence_or_create_new_visibility():
    engine = NavigationEngine(config())
    packet = observation(points=BOX_POINTS)
    engine.observe(packet, 0., [BOX])
    assert engine.layers.scan_count == 1
    engine.observe(packet, .1, [BOX])
    assert engine.layers.scan_count == 1
    assert engine.visible_at == {'box': 0.}


def test_mapping_reuses_live_crop_and_range_cut_without_changing_the_source(monkeypatch):
    from unittest.mock import Mock
    from deploy.robot import observation as packets

    cfg = room_navigation()
    engine = NavigationEngine(cfg)
    # The range-limit point is inside the field, while the high point is not in the height band.
    raw = observation(points=[[1., 0., .2], [4.95, 0., .2], [1., 0., .3]])
    snapshot = packets.Snapshot(raw, 1., 1., 1., None, (0., 0., 0.), 1)
    crop = Mock(wraps=packets.navigation_points)
    monkeypatch.setattr(packets, 'navigation_points', crop)
    view = snapshot.navigation_view(cfg['mapping']['point_height'])
    engine.observe(view.observation, 1.)
    engine.observe(view.observation, 1.1)
    crop.assert_called_once()
    lidar = view.observation['sensors']['lidar']
    assert engine.observation['sensors']['lidar']['pointcloud'] is lidar.measured_points
    np.testing.assert_array_equal(lidar.measured_points, [[1., 0., .2]])
    assert len(lidar['pointcloud']) == 2 and len(raw['sensors']['lidar']['pointcloud']) == 3


def test_navigation_emits_finite_bounded_velocity_and_retains_map_between_tasks():
    engine = NavigationEngine(config())
    engine.observe(observation(), 0.)
    layers = engine.layers
    engine.set_goal((1.5, 0., .4), 0.)
    commands = [engine.update(observation(i), i*.1) for i in range(4)]
    assert engine.phase == 'navigate'
    assert any(np.linalg.norm(command) > 0 for command in commands)
    for command in commands:
        assert command.shape == (3,) and np.isfinite(command).all()
        assert np.all(np.abs(command) <= np.array([.30, .15, .4])+1e-10)
    assert engine.limiter.policy_command is None
    before = layers.observed.copy()
    engine.skill_reports.append({'old_task': True})
    engine.set_goal((1., 1., .4), .5)
    assert engine.layers is layers
    np.testing.assert_array_equal(layers.observed, before)
    assert not engine.skill_reports


def test_unknown_target_selects_observed_frontier_without_marking_goal_free():
    engine = NavigationEngine(config())
    packet = observation()
    # Ground returns provide a free prefix; unknown space remains beyond it.
    packet['sensors']['lidar']['pointcloud'][:, 2] = 0.
    engine.observe(packet, 0.)
    target = (2.9, 0., .4)
    engine.set_goal(target, 0.)
    command = engine.update(packet, .1)
    assert engine.status == 'running'
    assert engine.decision['kind'] == 'explore'
    assert engine.goal_kind == 'frontier'
    assert not engine.layers.observed[engine.planner.cell(target)]
    assert np.isfinite(command).all()


def corridor_scan(step, x, doorway):
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    dx, dy = np.cos(angles), np.sin(angles)
    end_wall = 4.3 if doorway == 'open' else 2.5
    wall_distance_x = np.where(dx > 0, end_wall - x, x + 1.5)
    ranges = np.minimum(wall_distance_x / np.maximum(abs(dx), 1e-9),
                        1.05 / np.maximum(abs(dy), 1e-9))
    points = np.column_stack((x + ranges * dx, ranges * dy, np.full_like(dx, .25)))
    if doorway == 'slit':
        # The floor is visible beyond the gap, but its edges are too close for the body.
        slit = (points[:, 0] > 2.49) & (abs(points[:, 1]) < .21)
        points[slit, 0] = 4.05
        points[slit, 1] = (4.05 - x) * dy[slit] / dx[slit]
        points[slit, 2] = 0.
    packet = observation(step, position=(x, 0., .4), points=points)
    packet['sensors']['lidar']['position'] = np.array([x, 0., .25])
    return packet


@pytest.mark.parametrize('doorway', ['open', 'closed'])
def test_unseen_goal_follows_voronoi_to_an_observation_point_and_replans(doorway):
    engine = NavigationEngine(room_navigation())
    first = corridor_scan(0, 0., 'slit')
    goal = (3.5, 0., .4)
    engine.observe(first, 0.)
    observed = engine.layers.global_observed.copy()
    engine.set_goal(goal, .1)
    engine.update(first, .1)
    assert engine.goal_kind == 'frontier' and engine.decision['kind'] == 'explore'
    assert engine.routes.last_plan['engine'] == 'exploration_voronoi'
    planner = engine.planner
    start = planner.cell(first['position'])
    with engine.layers.global_planning() as boxes:
        blocked = planner.blocked(boxes)
        cells = [start, *(planner.cell(point) for point in engine.path)]
        assert len(cells) > 1 and cells[-1] == planner.cell(engine.endpoint)
        assert all(AStarMapPlanner.segment_clear(a, b, blocked) for a, b in zip(cells, cells[1:]))
    boundary = planner.cell(engine.decision['observation_boundary'])
    assert not observed[boundary]
    sight_blocked = ~observed | engine.layers.global_occupied
    sight_blocked[boundary] = False
    assert AStarMapPlanner.segment_clear(cells[-1], boundary, sight_blocked)
    assert not observed[planner.cell(goal)]
    np.testing.assert_array_equal(engine.layers.global_observed, observed)
    assert engine.update(first, .2)[0] > 0.

    # Reach the observation waypoint's arrival radius at the 30 cm route clearance.
    for step in range(1, 9):
        x = min(.3 * step, 1.95)
        engine.update(corridor_scan(step, x, doorway), step * .75)
    assert engine.goal == goal and engine.result is None
    if doorway == 'open':
        assert engine.layers.global_observed[planner.cell(goal)]
        assert engine.decision['kind'] == 'navigate' and engine.endpoint == goal[:2]
        assert engine.goal_kind == 'task' and engine.path
    else:
        assert engine.decision['kind'] == 'wait'
        assert not engine.layers.global_observed[planner.cell(goal)]
        assert not engine.path and not engine.layers.local_path


def test_deferred_observation_point_allows_another_view_of_the_same_opening():
    engine = NavigationEngine(room_navigation())
    start, goal = (0., 0.), (3.5, 0.)
    engine.set_goal(goal, 0.)
    engine.update(corridor_scan(0, 0., 'slit'), 0.)
    first = engine.endpoint
    engine.selector.defer_frontier(first, engine.now)
    decision, alternate = engine.selector.decide(engine.layers, start, goal)
    assert decision.kind == 'explore'
    assert np.linalg.norm(np.asarray(alternate) - first) >= .50
    assert decision.path and not engine.layers.global_observed[engine.planner.cell(goal)]


def test_goal_requires_measured_stopping_inside_release_bound():
    engine = NavigationEngine(config())
    engine.set_goal((0., 0., .4), 0.)
    for index in range(8):
        engine.update(observation(index, velocity=(.1, 0., 0.)), index*.1)
        assert engine.status == 'running'
    for index in range(8, 14):
        engine.update(observation(index), index*.1)
    assert engine.status == 'completed'
    assert engine.result['reason'] == 'goal_reached_and_stopped'


def test_translation_and_yaw_limits_are_independent_and_track_sent_command():
    engine = NavigationEngine(config())
    engine.observe(observation(), 0.)
    requested = np.array([.6, .15, 1.6])
    command = engine._output(requested)
    np.testing.assert_allclose(command, [.30, .075, .4])
    np.testing.assert_array_equal(engine.limiter.command, command)
    np.testing.assert_array_equal(requested, [.6, .15, 1.6])


@pytest.mark.parametrize('yaw', [-1.119007271, 1.119007271])
def test_recorded_turn_does_not_reduce_translation_in_navigation_adapter(yaw):
    engine = NavigationEngine(config())
    engine.observe(observation(), 0.)
    requested = [.111664123, -.08, yaw]
    command = engine._output(requested)
    np.testing.assert_allclose(command[:2], requested[:2])
    assert command[2] == pytest.approx(np.sign(yaw)*.4)


def test_limiter_checks_bounded_velocity_before_acceptance_with_independent_acceleration():
    engine = NavigationEngine(config())
    previous = np.zeros(3)
    engine.limiter.accept(previous, 0.)
    for index in range(1, 31):
        command = engine.limiter.preview([.30, 0., 1.6], index*.1)
        assert np.linalg.norm(command[:2]-previous[:2]) <= .05+1e-9
        assert abs(command[2]-previous[2]) <= .15+1e-9
        assert abs(command[2]) <= .4
        engine.limiter.accept(command, index*.1)
        previous = command
    np.testing.assert_allclose(command, [.30, 0., .4])


def test_clipped_arc_is_rechecked_and_stops_if_blocked(monkeypatch):
    from deploy.app import navigation
    engine = NavigationEngine(config())
    engine.observe(observation(), 0.)
    checked = []
    def blocked(planner, boxes, position, heading, forward, yaw, **kwargs):
        checked.append((forward, kwargs['lateral'], yaw))
        return False
    monkeypatch.setattr(navigation, 'body_motion_clear', blocked)
    command = engine._output([.6, .15, 1.6])
    np.testing.assert_allclose(checked, [[.3, .075, .4]])
    np.testing.assert_array_equal(command, np.zeros(3))
    np.testing.assert_array_equal(engine.limiter.command, command)
    assert engine.events[-1]['event'] == 'velocity_limit_blocked'


def test_pause_discards_contact_state_and_observe_cannot_resume_it():
    engine = NavigationEngine(config())
    engine.observe(observation(), 0.)
    engine.set_goal((1., 0., .4), 0.)
    engine.phase = 'skill'
    engine.skill = SimpleNamespace(state='push')
    engine.pause()
    retained = engine.layers
    engine.observe(observation(1), .1)
    assert engine.skill is None and engine.status == 'idle' and engine.phase == 'paused'
    engine.set_goal((1., 0., .4), .2)
    assert engine.phase == 'decide' and engine.layers is retained


@pytest.mark.parametrize('operation', ['set_goal', 'cancel'])
def test_new_perception_epoch_retires_old_ids_without_freeing_their_cells(operation):
    engine = NavigationEngine(config(interaction_enabled=True))
    engine.observe(observation(points=BOX_POINTS), 1., [BOX])
    layers = engine.layers
    cell = engine.planner.cell(BOX.xy)
    assert layers.observed[cell] and layers.tracked_occupied[cell]
    if operation == 'set_goal':
        engine.set_goal((2., 0., .4), 1.1)
    else:
        engine.cancel()
    assert not layers.detected_boxes and not layers.current_boxes and not layers.global_boxes
    assert not layers.tracked_occupied.any() and not layers.global_tracked.any()
    assert not layers.observed[cell] and not layers.global_observed[cell]
    assert not engine.planner.free[cell]
    assert not engine.planner.interactable_ids
    assert engine.layers is layers


def test_stale_track_grace_preserves_occupancy_then_retires_to_unknown():
    engine = NavigationEngine(config())
    engine.observe(observation(points=BOX_POINTS), 1., [BOX])
    cell = engine.planner.cell(BOX.xy)
    engine.observe(observation(1, points=[(-2., 0., .4)]), 1.8, [])
    assert 'box' in engine.layers.detected_boxes
    assert engine.layers.local_occupied[cell]
    engine.observe(observation(2, points=[(-2., 0., .4)]), 2.1, [])
    assert 'box' not in engine.layers.detected_boxes
    assert not engine.layers.observed[cell]
    assert not engine.planner.free[cell]


def test_box_approach_weak_hits_block_locally_before_changing_known_global_free_space():
    from deploy.app.config import ROOT
    settings = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(settings['navigation'], settings['interaction'])
    layers = engine.layers
    box = BoxState('box-1', (1., 0.), (.3, .8, .3))
    engine.observe(observation(points=[(.86, dy, .20) for dy in (-.08, 0., .08)]), 1., [box])
    known, new, far, other = map(engine.planner.cell, [(.6, -.4), (.6, -.5), (3., -.4), (.6, -.6)])
    # Seed two known approach cells and one equally clear cell far from the box.
    for cell in (known, far, other):
        layers.observed[cell] = True
        layers.evidence.log_odds[cell] = -.1
    assert not layers.observed[new]
    layers.refresh_global(reason='field_fixture')
    hit = np.zeros_like(layers.observed)
    for cell in (known, new, far):
        hit[cell] = True
    free = np.zeros_like(hit)
    for count in (1, 2, 3):
        layers.ingest_scan_masks(hit, free)
        layers.refresh_global(reason='measured_scan')
        assert layers.local_occupied[known]
        assert layers.global_occupied[known] == (count == 3)
        assert layers.global_occupied[new] and layers.global_occupied[far]
        assert layers.global_occupied[engine.planner.cell(box.xy)]
    # A confirmed obstacle uses the ordinary clearing threshold.
    free[known] = True
    layers.ingest_scan_masks(np.zeros_like(hit), free)
    layers.refresh_global(reason='measured_free')
    assert layers.global_occupied[known]
    engine.push_speed_sent(box.id)
    hit[:] = False
    hit[other] = True
    layers.ingest_scan_masks(hit, free)
    layers.refresh_global(reason='after_push')
    assert layers.global_occupied[other]


def test_registered_box_jitter_retains_coverage_until_that_box_is_pushed():
    registry = [{'id': key, 'push_allowed': True} for key in ('box', 'other')]
    engine = NavigationEngine(config(), {'source': 'registered_box_lidar', 'boxes': registry})
    layers = engine.layers
    # Measure faces from the front. The far edge receives no free rays.
    def observe_pair(step, offset):
        boxes = [BoxState(row['id'], (1. + offset, y), (.4, .4, .5))
                 for row, y in zip(registry, (0., 1.))]
        points = [(box.xy[0] - .2, box.xy[1] + dy, .3)
                  for box in boxes for dy in (-.05, 0., .05)]
        engine.observe(observation(step, points=points), float(step), boxes)
        return [engine.planner.box_mask(box) for box in boxes]

    old = observe_pair(1, 0.)
    new = observe_pair(2, -.1)
    for before, after in zip(old, new):
        edge = before & ~after
        assert edge.any() and layers.global_observed[edge].all()
        assert layers.observed[edge].all() and not layers.tracked_occupied[edge].any()
        assert layers.global_occupied[after].all()
    assert not any(e['event'] == 'visible_box_moved' for e in layers.events)
    assert not layers.observed[engine.planner.cell((1.45, 0.))]

    engine.push_speed_sent('box')
    engine.set_goal((2., 0.), 2.1)
    after_push = observe_pair(3, -.2)
    pushed_edge = new[0] & ~after_push[0]
    stationary_edge = new[1] & ~after_push[1]
    assert pushed_edge.any() and not layers.global_observed[pushed_edge].any()
    assert stationary_edge.any() and layers.global_observed[stationary_edge].all()
    assert layers.global_occupied[after_push[0]].all()


def test_expired_identity_cannot_erase_overlapping_new_live_geometry():
    engine = NavigationEngine(config())
    engine.observe(observation(points=BOX_POINTS), 1., [BOX])
    replacement = BoxState('new-id', (1.1, 0.), (.4, .4, .5))
    points = [(.9, -.05, .3), (.9, 0., .3), (.9, .05, .3)]
    engine.observe(observation(1, points=points), 1.8, [replacement])
    assert set(engine.layers.detected_boxes) == {'box', 'new-id'}
    engine.observe(observation(2, points=points), 2.1, [replacement])
    assert set(engine.layers.detected_boxes) == {'new-id'}
    overlap = engine.planner.cell((1., 0.))
    assert engine.layers.observed[overlap] and engine.layers.local_occupied[overlap]
    vacated = engine.planner.cell((.82, 0.))
    assert not engine.layers.observed[vacated]
    assert not engine.planner.free[vacated]


def test_global_clearance_buffer_uses_release_recovery_before_task_decision():
    engine = NavigationEngine(config(route_clearance=.4))
    layers = engine.layers
    rows, _ = np.indices(layers.observed.shape)
    wall = rows >= engine.planner.cell((0., .35))[0]
    layers.ingest_scan_masks(wall, ~wall)
    layers._latest_static = layers.evidence.occupied_mask()
    layers.refresh_global()
    layers._bind_local()
    engine.set_goal((1.5, -.7, .4), 0.)
    command = engine.update(observation(points=[]), 0.)
    assert engine.decision['kind'] == 'recover'
    assert engine.recovery_pending_settle
    assert engine.status == 'running' and np.isfinite(command).all()


def test_rising_tilt_requests_zero_before_hard_failure_limit():
    engine = NavigationEngine(config())
    engine.set_goal((1.5, 0., .4), 0.)
    packet = observation()
    angle = np.deg2rad(20.)/2
    packet['orientation'] = np.array([np.cos(angle), 0., np.sin(angle), 0.])
    command = engine.update(packet, 0.)
    np.testing.assert_array_equal(command, np.zeros(3))
    assert engine.status == 'running'
    assert engine.decision['execution_state'] == 'attitude_recovery_hold'
    assert engine.attitude.minimum_hold_ticks == 8
    assert engine.attitude.maximum_hold_ticks == 60
