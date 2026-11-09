"""Measured-ray geometry, independent of quantized angular channels."""
import numpy as np
import pytest

from grutopia_extension.interactive_navigation.lidar_raycast import clearing_endpoints, ray_cells
from grutopia_extension.interactive_navigation.mapping import MappingConfig, OccupancyGridMap


def plane_intersection_cells(origin, endpoints, grid_origin, cell_size, lower, upper):
    """Independent reference: split segments at every plane, inspect midpoints."""
    origin, grid_origin, cell_size, lower, upper = map(
        np.asarray, (origin, grid_origin, cell_size, lower, upper))
    found = set()
    for endpoint in endpoints:
        delta = np.asarray(endpoint)-origin
        if not np.any(delta):
            continue
        fractions = [0., 1.]
        for axis in range(3):
            if delta[axis] == 0:
                continue
            lo = int(np.floor((lower[axis]-grid_origin[axis])/cell_size[axis]))
            hi = int(np.ceil((upper[axis]-grid_origin[axis])/cell_size[axis]))
            planes = np.r_[lower[axis], upper[axis],
                           grid_origin[axis]+np.arange(lo, hi+1)*cell_size[axis]]
            fractions.extend(t for t in (planes-origin[axis])/delta[axis] if 0<t<1)
        fractions = np.unique(fractions)
        for left, right in zip(fractions[:-1], fractions[1:]):
            if right-left < 1e-11:
                continue
            middle = origin+((left+right)/2)*delta
            if np.all(middle >= lower) and np.all(middle < upper):
                found.add(tuple(np.floor((middle-grid_origin)/cell_size).astype(int)))
    return found


@pytest.mark.parametrize('origin,endpoints', [
    ((0., 0., 0.), [(1., 1., 1.)]),  # exact grid corners; no side-touch cells
    ((1., 1., 1.), [(0., 0., 0.)]),  # negative travel from upper face
    ((-.5, .35, .4), [(1.5, .35, .4)]),  # clipped entry and exit
    ((.25, .35, .4), [(.25, .95, .4), (.25, -.4, .4)]),  # zero X/Z axes
    ((.25, .35, .4), [(.25, .35, .4)]),  # zero-length is no visibility
    ((.25, .35, .4), [(.25, .35, .4), (.85, .35, .4)]),
    ((1., .35, .4), [(1., .85, .4)]),  # parallel along excluded upper face
    ((0., .35, .4), [(0., .85, .4)]),  # included lower face
    ((.25, .35, -.5), [(.25, .35, .6)]),  # non-grid-aligned height clipping
])
def test_ray_traversal_matches_independent_plane_intersections(origin, endpoints):
    args = (np.array(origin), np.asarray(endpoints), np.array([0.,0.,.15]),
            np.array([.1,.1,.1]), np.array([0.,0.,-.07]), np.array([1.,1.,.63]))
    expected = plane_intersection_cells(*args)
    actual = {tuple(cell) for cell in ray_cells(*args)}
    assert actual == expected


def test_random_nonuniform_grids_and_clipped_returns_match_geometry():
    rng = np.random.default_rng(4185)
    for _ in range(60):
        origin = rng.uniform(-1., 1., 3)
        endpoints = rng.uniform(-1.2, 1.2, (8, 3))
        grid_origin = np.array([-.7,-.7,.15])
        cell_size = np.array([.1,.1,.07])
        lower, upper = np.array([-.7,-.7,-.05]), np.array([.8,.8,.63])
        args = (origin, endpoints, grid_origin, cell_size, lower, upper)
        assert {tuple(cell) for cell in ray_cells(*args)} == plane_intersection_cells(*args)


def test_duplicate_beams_keep_nearer_return_but_distinct_heights_survive():
    origin = np.array([.05,.05,.4])
    points = np.array([[1.05,.05,.4], [4.05,.05,.4], [1.05,.05,.435], [4.05,.05,.35]])
    selected = {tuple(p) for p in clearing_endpoints(origin, points, 900)}
    assert selected == {tuple(points[i]) for i in (0,2,3)}


def test_rotating_budget_eventually_visits_every_distinct_beam():
    origin = np.zeros(3)
    points = np.column_stack((np.ones(17), np.linspace(-.5,.5,17), np.full(17,.3)))
    covered = set()
    for phase in range(len(points)):
        selected = clearing_endpoints(origin, points, 3, phase)
        assert len(selected) == 3
        covered.update(map(tuple,selected))
    assert covered == set(map(tuple,points))


def grid():
    return OccupancyGridMap(MappingConfig(
        x_limits=(0.,6.), y_limits=(-1.,1.), obstacle_height=(.15,.63),
        lidar_height_resolution=.1, sticky_free_scale=.25,
        obstacle_inflation_radius=.4))


def scans(g, points, origin=(.05,.05,.4), count=1):
    for _ in range(count):
        g.update_lidar(origin,np.asarray(points),8.)


def test_different_beam_in_same_former_elevation_group_clears_removed_obstacle():
    g = grid()
    old = g.world_to_cell((2.05,.05))
    goal = g.world_to_cell((2.15,.05))
    scans(g,[[2.05,.05,.375]],count=3)
    assert g.occupied_mask()[old] and g.inflated_mask()[goal]
    # Both angles round to 0 degrees in the former 5-degree grouping. The
    # distant beam passes exactly through the old point; nearer beam does not.
    scans(g,[[1.05,.05,.435],[4.05,.05,.35]],count=60)
    assert g.observed[goal]
    assert not g.occupied_mask()[old]
    assert not g.inflated_mask()[goal]


def test_ray_does_not_clear_wrong_height_from_cell_index_interpolation():
    g = grid()
    old = g.world_to_cell((.35,.05))
    for _ in range(3):
        g.update_obstacle_points(np.array([[.35,.05,.3],[.35,.05,.375]]))
    lower = g._height_bin(.3)
    upper = g._height_bin(.375)
    old_lower = g._height_evidence[(lower,*old)]
    scans(g,[[2.6535967432970597,.46012050777724345,.11363344759915474]],
          origin=(.04609852680672011,.05742341404129132,.38040238938711585),count=60)
    # The ray crosses this XY cell only above z=.35, although the old
    # Bresenham/index interpolation put it below z=.35.
    assert g._height_evidence[(lower,*old)] == old_lower
    assert g._height_evidence[(upper,*old)] == 0
    assert g.occupied_mask()[old]


def test_unseen_and_occluded_obstacles_survive_far_duplicate_returns():
    g = grid()
    occluded = g.world_to_cell((2.05,.05))
    unseen = g.world_to_cell((2.05,.65))
    scans(g,[[2.05,.05,.4],[2.05,.65,.4]],count=3)
    before = g.log_odds.copy()
    scans(g,[[1.05,.05,.4],[4.05,.05,.4]],count=60)
    assert g.log_odds[occluded] == before[occluded]
    assert g.log_odds[unseen] == before[unseen]


def test_return_outside_map_only_clears_measured_segment_inside_map():
    g = grid()
    scans(g,[[5.55,.05,.4]],count=3)
    old = g.world_to_cell((5.55,.05))
    # Valid measured endpoint is beyond bounded map; its in-map segment is
    # still actual free evidence, with no fabricated beyond-range extension.
    scans(g,[[6.55,.05,.4]],count=60)
    assert g.observed[old] and not g.occupied_mask()[old]
