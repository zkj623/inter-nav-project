"""Batching must preserve every collision verdict of the scalar 19-pose sweep."""
import math

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.body_geometry import BodyFootprint
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_motion_clear, body_rotation_clear, navigation_footprint_samples,
)
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def scalar_sweep(planner, boxes, positions, angles, margin):
    raw = planner.free.copy()
    for box in boxes:
        mask = planner.box_mask(box)
        if mask is None:
            return False
        raw &= ~mask
    xx, yy = navigation_footprint_samples(planner, margin)
    for position, angle in zip(positions, angles):
        row, col = planner.cell(position)
        c, s = math.cos(angle), math.sin(angle)
        cols = col + np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        rows = row + np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        if rows.min() < 0 or cols.min() < 0 or rows.max() >= raw.shape[0] or cols.max() >= raw.shape[1]:
            return False
        if not raw[rows, cols].all():
            return False
    return True


@pytest.mark.parametrize('size,custom_body', [(41, False), (81, True), (161, False)])
def test_motion_and_turn_match_scalar_reference_across_obstacles_and_grid_edges(size, custom_body):
    rng = np.random.default_rng(20261117+size)
    planner = SidePushPlanner(np.ones((size, size), bool), [[-2., 2.], [-2., 2.]],
                              resolution=4./(size-1))
    if custom_body:
        planner.body_geometry = BodyFootprint(-.37, .56, .24)
    verdicts = set()
    for index in range(250):
        planner.free[:] = True
        # Sparse measured/unknown blocked cells leave both clear and blocked
        # trajectories. Their collision meaning is identical to the planner.
        if index % 3:
            cells = rng.integers(0, size, size=(30, 2))
            planner.free[cells[:, 0], cells[:, 1]] = False
        position = rng.uniform(-1.8, 1.8, 2)
        if index % 4 == 0:
            position = np.rint(position/planner.resolution)*planner.resolution
            position += (0., 1e-14, -1e-14)[index % 3]
        heading = float(rng.uniform(-math.pi, math.pi))
        forward, lateral = rng.uniform(-.8, .8, 2)
        rotation = (0., 1e-9, -1e-8, 1.0001e-8, float(rng.uniform(-1.5, 1.5)))[index % 5]
        horizon = (0., .25, .8, 1.6)[index % 4]
        margin = (0., .025)[index % 2]
        boxes = [] if index % 3 else [BoxState('box', (.7, -.4), (.35, .3, .5), 25.)]
        times = np.linspace(0., horizon, 19)
        positions, angles = [], []
        for t in times:
            angle = heading+rotation*t
            if abs(rotation) > 1e-8:
                delta = forward/rotation*np.array([math.sin(angle)-math.sin(heading), math.cos(heading)-math.cos(angle)])
                delta += lateral/rotation*np.array([math.cos(angle)-math.cos(heading), math.sin(angle)-math.sin(heading)])
            else:
                delta = t*np.array([forward*math.cos(heading)-lateral*math.sin(heading),
                                    forward*math.sin(heading)+lateral*math.cos(heading)])
            positions.append(position+delta)
            angles.append(angle)
        expected = scalar_sweep(planner, boxes, positions, angles, margin)
        verdicts.add(expected)
        assert body_motion_clear(planner, boxes, position, heading, forward, rotation,
                                 horizon, lateral, margin=margin) == expected, index
        end_heading = heading+float(rng.uniform(-math.pi, math.pi))
        expected_turn = scalar_sweep(planner, boxes, [position]*19,
                                     np.linspace(heading, end_heading, 19), margin)
        assert body_rotation_clear(planner, boxes, position, heading, end_heading, margin) == expected_turn, index
    assert verdicts == {False, True}


def test_cached_body_samples_follow_geometry_changes_and_never_cache_map_clearance():
    planner = SidePushPlanner(np.ones((81, 81), bool), [[-2., 2.], [-2., 2.]], resolution=.05)
    assert body_motion_clear(planner, [], (0., 0.), 0., 0., 0.)
    planner.free[planner.cell((.4, 0.))] = False
    assert not body_motion_clear(planner, [], (0., 0.), 0., 0., 0.)
    planner.free[:] = True
    planner.body_geometry = BodyFootprint(-.32, .5, .22)
    assert body_motion_clear(planner, [], (1.2, 0.), 0., 0., 0.)
    planner.body_geometry = BodyFootprint(-.32, 1., .22)
    assert not body_motion_clear(planner, [], (1.2, 0.), 0., 0., 0.)
