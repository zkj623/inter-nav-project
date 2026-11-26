"""Cold control work and ordinary frontier probes must not force routine stops."""
import numpy as np
import pytest

from deploy.app.config import load
from deploy.app.navigation import NavigationEngine, PlanningRequired
from deploy.tests.test_planning_handoff import run
from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision, BoxState


def engine():
    config = load()['navigation']
    config['mapping'].update(x_limits=[-3., 3.], y_limits=[-3., 3.], grid_resolution=.1)
    result = NavigationEngine(config)
    result.set_goal((1.8, 1.8, .4), 10.)
    result.position = np.array([-1.8, -1.8, .4])
    result.heading = 0.
    return result


def test_local_turn_placement_search_waits_for_the_motion_writer(run, monkeypatch):
    from grutopia_extension.interactive_navigation import local_turn_maneuver as turn

    engine = run.tasks.engine
    selector = turn.LocalTurnManeuver(.8)
    checks = []

    def rotation_clear(*args, **kwargs):
        checks.append(run.motion.state)
        return True

    monkeypatch.setattr(turn, 'body_rotation_clear', rotation_clear)
    monkeypatch.setattr(turn, 'body_translation_recovery_clear', lambda *args: True)
    assert run.motion.state == 'active'
    with pytest.raises(PlanningRequired, match='local_turn_search'):
        selector._select_stage(engine.planner, (), engine.position[:2], engine.heading)
    assert not checks
    run.feedback('stopped', 'p')
    engine.planner.free = np.ones_like(engine.planner.free)
    assert selector._select_stage(engine.planner, (), engine.position[:2], engine.heading) is not None
    assert set(checks) == {'stopped'}


def test_connectivity_probe_matches_full_decision_and_preserves_live_map():
    value = engine()
    rng = np.random.default_rng(20261117)
    outcomes = set()
    for index in range(40):
        layers = value.layers
        layers.global_observed[:] = True
        layers.static_occupied[:] = False
        if index % 4:
            cells = rng.integers(0, layers.observed.shape[0], (index*2, 2))
            layers.static_occupied[cells[:, 0], cells[:, 1]] = True
        layers._bind_local()
        local_free = value.planner.free
        expected = value.selector.task_decision(layers, value.position[:2], value.goal[:2]) is not None
        actual = value._task_route_available()
        assert actual == expected
        assert value.planner.free is local_free and value.planner.planning_scope == 'local'
        outcomes.add(actual)
    assert outcomes == {False, True}


def test_frontier_probe_keeps_unknown_and_diagonal_corners_blocked():
    value = engine()
    layers = value.layers
    value.position = np.array([0., 0., .4])
    value.goal = (.1, .1, .4)
    start, goal = value.planner.cell(value.position), value.planner.cell(value.goal)
    layers.global_observed[:] = False
    layers.global_observed[start] = layers.global_observed[goal] = True
    layers.static_occupied[:] = False
    assert not value._task_route_available()
    layers.global_observed[start[0], goal[1]] = True
    layers.global_observed[goal[0], start[1]] = True
    assert value._task_route_available()
    layers.global_observed[goal] = False
    assert not value._task_route_available()


def test_known_box_does_not_interrupt_exploration_toward_an_unseen_or_disconnected_goal():
    value = engine()
    value.layers.global_boxes = [BoxState('box', (0., 0.), (.4, .4, .5))]
    value.planner.interactable_ids = {'box'}
    value.before_planning = lambda reason: pytest.fail(reason)
    assert not value._task_route_available()
    for xy in (value.position, value.goal):
        value.layers.global_observed[value.planner.cell(xy)] = True
    assert not value._task_route_available()
    value.layers.global_observed[:] = True
    assert value._task_route_available()


def test_only_a_potential_push_yields_and_failed_push_is_retried_on_new_evidence(monkeypatch):
    value = engine()
    value.position = np.array([-1.5, 0., .4])
    value.goal = (1.5, 0., .4)
    layers = value.layers
    row, col = value.planner.cell((0., 0.))
    layers.global_observed[:] = False
    layers.global_observed[row-1:row+2, col-16:col+17] = True
    layers.global_boxes = [BoxState('box', (0., 0.), (.4, .4, .5))]
    value.before_planning = lambda reason: pytest.fail(reason)
    assert not value._task_route_available()  # The box cannot be moved yet.
    value.planner.interactable_ids = {'box'}
    def moving(reason):
        raise PlanningRequired(reason)
    value.before_planning = moving
    with pytest.raises(PlanningRequired, match='frontier_reconsideration'):
        value._task_route_available()
    value.before_planning = None
    monkeypatch.setattr(value.selector, 'task_decision', lambda *args: None)
    assert not value._task_route_available()
    value.before_planning = moving
    value.now += 1.
    assert not value._task_route_available()
    layers.global_revision += 1
    with pytest.raises(PlanningRequired, match='frontier_reconsideration'):
        value._task_route_available()
    value.before_planning = None
    monkeypatch.setattr(value.selector, 'task_decision', lambda *args: object())
    assert value._task_route_available()


def test_initial_route_has_local_control_prepared_without_seeding_unsent_speed(run):
    engine = run.tasks.engine
    assert engine._prepared_tracker is engine.tracker
    assert engine.last_tracking is not None
    assert not run.motion.commands
    np.testing.assert_array_equal(engine.limiter.command, np.zeros(3))
    assert engine.next_reconsider > engine.now


@pytest.mark.parametrize('available', ['next_frontier', 'same_frontier_other_path', 'none'])
def test_stopped_writer_requires_an_executable_frontier_before_resuming(run, monkeypatch, available):
    engine = run.tasks.engine
    goal = engine.goal
    first, second = (.8, -.5), (.8, 0.)
    choices = [first, second] if available == 'next_frontier' else [first]
    attempts = []

    def select(layers, start, goal):
        for target in choices:
            if engine.selector.key(target) not in engine.selector.visited:
                return BoxDecision('explore', 'goal_unseen', path=(target,)), target
        return BoxDecision('wait', 'no_reachable_observed_frontier'), None

    def route(target):
        path = ((.35, -.3), target) if available == 'same_frontier_other_path' else (target,)
        engine.layers.global_path = path
        return path

    original = engine._tracker

    def tracker(path, *, short_goal):
        result = original(path, short_goal=short_goal)
        target = tuple(path[-1][:2])
        blocked = target == first and (available != 'same_frontier_other_path' or len(path) > 1)

        def action(*args):
            attempts.append((target, run.motion.state))
            if blocked:
                return {'move_by_speed': [0., 0., 0.]}, {
                    'needs_replan': True, 'sharp_turn_space_unavailable': True,
                    'observation_timeout_or_blocked': True, 'blocked_ticks': 1}
            return {'move_by_speed': [.04, 0., 0.]}, {'body_motion_safe': True}
        result.action = action
        return result

    monkeypatch.setattr(engine.selector, 'decide', select)
    monkeypatch.setattr(engine, '_route', route)
    monkeypatch.setattr(engine, '_tracker', tracker)
    monkeypatch.setattr(engine.planner, 'path', lambda start, target, boxes: (tuple(target),))
    token = run.hold()
    run.feedback('stopped', token)
    run.controller.step(run.sample(), run.clock[0])
    assert run.motion.arm_count == 1 and not run.motion.commands
    assert attempts and all(state == 'stopped' for _, state in attempts)
    assert engine.goal == goal
    if available == 'none':
        assert run.controller.planning_hold is None and run.tasks.state == 'paused'
        assert engine.decision['kind'] == 'wait'
        return
    selected = second if available == 'next_frontier' else first
    assert engine.endpoint == selected
    assert (engine.selector.key(first) in engine.selector.visited) == (selected == second)
    np.testing.assert_array_equal(engine.limiter.command, np.zeros(3))
    run.controller.step(run.sample(), run.clock[0])
    assert run.motion.arm_count == 2 and not run.motion.commands
    run.controller.step(run.sample(), run.clock[0])
    assert run.controller.planning_hold is None and run.motion.stops == [token]
    assert run.motion.commands[-1][0][0] > 0


def test_safe_observation_and_recovery_keep_the_frontier():
    value = engine()
    value.goal_kind, value.endpoint = 'frontier', (.6, -.6)
    for detail in [
            {'sharp_turn_space_unavailable': True, 'observation_motion': True, 'needs_replan': False},
            {'needs_replan': True, 'observation_timeout_or_blocked': True, 'recovering_buffer': True},
            {'needs_replan': False, 'staging_phase': 'staging_brake'}]:
        assert not value._reject_failed_frontier(detail)
    assert not value.selector.visited


def test_worker_timeout_remains_visible_and_normal_completion_is_not_an_error(run):
    run.feedback('stopping', 'command_timeout')
    run.feedback('stopped', 'command_timeout')
    current = run.sample()
    assert run.controller.step(current, run.clock[0]) == 'command_timeout'
    run.controller.task_error = None
    run.tasks.state = 'completed'
    run.feedback('stopping', 'task_completed')
    run.feedback('stopped', 'task_completed')
    assert run.controller.task_error is None
