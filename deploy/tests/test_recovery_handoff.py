"""Body-buffer recovery must resume the exit selected while stopped.

Live run, 2026-11-18: an unconditional planning boundary before the recovery
call stopped the robot again on every resume (holds 29..62 back to back), so
the cached exit target was never executed while moving.
"""
import numpy as np
import pytest

from deploy.app.navigation import PlanningRequired
from deploy.app.tasks import Tasks
from deploy.robot.observation import Snapshot
from deploy.tests.test_online_navigation import config, observation


def scene():
    tasks = Tasks(config(route_clearance=.4, tick_hz=10., task_timeout_s=300.))
    engine, layers = tasks.engine, tasks.engine.layers
    rows, _ = np.indices(layers.observed.shape)
    occupied = rows >= engine.planner.cell((0., .35))[0]
    layers.ingest_scan_masks(occupied, ~occupied)
    layers._latest_static = layers.evidence.occupied_mask()
    layers.refresh_global()
    layers._bind_local()
    tasks.map_ready = True
    tasks.start('navigate', (1.5, -.7, .4), 0.)
    return tasks


def sample(now, *, velocity=(0., 0., 0.)):
    packet = observation(int(round(now*100)), velocity=velocity, points=[])
    return Snapshot(packet, now, now, now, None, velocity, int(round(now*100)))


def reject_planning(reason):
    raise PlanningRequired(reason)


def test_global_recovery_only_yields_before_new_candidate_search():
    tasks = scene()
    engine = tasks.engine
    engine.before_planning = reject_planning
    # Moving, with no selected exit: the search itself requires a stop, and
    # the boundary precedes any mutation of the cached target.
    with pytest.raises(PlanningRequired, match='decision_buffer_recovery'):
        tasks.update(sample(0.), 0.)
    assert engine.planner._global_buffer_exit is None

    engine.before_planning = None
    assert tasks.prepare_for_motion(sample(.1), .1)
    target = dict(engine.planner._global_buffer_exit)
    assert engine.phase == 'decide'
    tasks.activate_prepared(.1)
    engine.before_planning = reject_planning
    # Each tick must recheck the cached exit and produce a command, rather
    # than unconditionally disarming again before consulting that target.
    for now in (.2, .3, .4):
        command = tasks.update(sample(now), now)
        assert np.linalg.norm(command) > 0
        assert engine.last_tracking['reused_exit_target']
        assert engine.planner._global_buffer_exit == target


def scan(now, points=None):
    packet = observation(int(round(now*100)), points=points)
    return Snapshot(packet, now, now, now, None, (0., 0., 0.), int(round(now*100)))


def test_local_only_obstacle_at_route_clearance_exits_buffer_instead_of_replanning():
    """Live run, 2026-11-18: a return 0.38 m to the right reached the 10 Hz
    local map before the 1 Hz global map. The tracker checks route clearance
    on the local map and refused to start from the robot cell; the decision
    checks the global map, saw no buffer, and re-issued the same route."""
    tasks = Tasks(config(route_clearance=.4, tick_hz=10., task_timeout_s=300.))
    engine, layers = tasks.engine, tasks.engine.layers
    for step in range(11):
        tasks.update(scan(step*.1), step*.1)
    ring = observation()['sensors']['lidar']['pointcloud']
    strip = np.array([(x, -.38, .4) for x in np.arange(-.3, 1.2, .05)])
    both = np.vstack([ring, strip])
    tasks.update(scan(1.05, both), 1.05)
    tasks.start('navigate', (1.5, 0., .4), 1.05)
    assert tasks.prepare_for_motion(scan(1.1, both), 1.1)
    cell = engine.planner.cell((0., 0.))
    with layers.global_planning() as global_boxes:
        assert not engine.planner.blocked(global_boxes)[cell]
    assert engine._tracker_route_start_blocked(layers.current_boxes)
    assert engine.decision['kind'] == 'recover'
    tasks.activate_prepared(1.1)
    engine.before_planning = reject_planning
    command = tasks.update(scan(1.2, both), 1.2)
    assert np.linalg.norm(command) > 0
    assert engine.last_tracking['reused_exit_target']
