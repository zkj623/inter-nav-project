"""Cold control work and ordinary frontier probes must not force routine stops."""
import numpy as np
import pytest

from deploy.app.config import load
from deploy.app.navigation import NavigationEngine, PlanningRequired
from deploy.tests.test_planning_handoff import run
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


def engine():
    config = load()['navigation']
    config['mapping'].update(x_limits=[-3., 3.], y_limits=[-3., 3.], grid_resolution=.1)
    result = NavigationEngine(config)
    result.set_goal((1.8, 1.8, .4), 10.)
    result.position = np.array([-1.8, -1.8, .4])
    result.heading = 0.
    return result


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


def test_box_reconsideration_still_yields_before_full_interaction_planning():
    value = engine()
    value.layers.global_boxes = [BoxState('box', (0., 0.), (.4, .4, .5))]
    def boundary(reason):
        raise PlanningRequired(reason)
    value.before_planning = boundary
    with pytest.raises(PlanningRequired, match='frontier_reconsideration'):
        value._task_route_available()


def test_initial_route_has_local_control_prepared_without_seeding_unsent_speed(run):
    engine = run.tasks.engine
    assert engine._prepared_tracker is engine.tracker
    assert engine.last_tracking is not None
    assert not run.motion.commands
    np.testing.assert_array_equal(engine.limiter.command, np.zeros(3))
    assert engine.next_reconsider > engine.now


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
