"""Measured-height obstacle support with the existing planar navigation map."""
import numpy as np
import pytest

from grutopia_extension.interactive_navigation.mapping import FusedMap, MappingConfig, OccupancyGridMap


def grid(**changes):
    config = dict(x_limits=(0., 6.), y_limits=(-1., 1.), obstacle_height=(.15, 1.6),
                  sticky_free_scale=.25, lidar_height_resolution=.1)
    config.update(changes)
    return OccupancyGridMap(MappingConfig(**config))


def scan(g, points, origin=(.05, .05, .4), repeats=1):
    for _ in range(repeats):
        g.update_lidar(origin, np.asarray(points, dtype=float), 8.)


def test_ground_observes_empty_space_without_erasing_tabletop():
    g = grid()
    tabletop = g.world_to_cell((1.05, .05))
    empty = g.world_to_cell((.55, .05))
    scan(g, [[1.05, .05, .8]], repeats=3)
    support = g.log_odds[tabletop]
    scan(g, [[3.05, .05, 0.]], repeats=60)
    assert g.log_odds[tabletop] == support
    assert g.occupied_mask()[tabletop]
    assert g.observed[empty] and not g.occupied_mask()[empty]
    assert g.log_odds[empty] < 0


def test_go2_low_box_survives_gap_rays_with_table_above_collision_height():
    g = grid(obstacle_height=(.15, .63))
    box = g.world_to_cell((1.05, .05))
    scan(g, [[1.05, .05, .35]], repeats=3)
    support = float(g.log_odds[box])
    # A .50 m gap passes over the box; the .80 m tabletop is outside the
    # navigation collision band. Neither is evidence that the box disappeared.
    scan(g, [[3.05, .05, .50], [1.05, .05, .80]],
         origin=(.05, .05, .50), repeats=60)
    assert g.occupied_mask()[box]
    assert float(g.log_odds[box]) == support
    # Once the box is removed, actual rays at its own height can clear it,
    # even while the irrelevant tabletop remains visible.
    scan(g, [[3.05, .05, .35], [1.05, .05, .80]],
         origin=(.05, .05, .35), repeats=60)
    assert g.observed[box] and not g.occupied_mask()[box]


def test_high_tabletop_remains_in_scene_cloud_but_not_navigation_occupancy():
    config = MappingConfig(x_limits=(0., 6.), y_limits=(-1., 1.),
                           obstacle_height=(.15, .63), point_height=(-.1, 2.),
                           lidar_height_resolution=.1)
    mapping = FusedMap(config)
    tabletop = np.array([[1.05, .05, .8]])
    mapping.update_lidar((.05, .05, .4), tabletop, step=0, max_range=8.)
    np.testing.assert_allclose(mapping.voxels.points(), tabletop)
    cell = mapping.occupancy.world_to_cell(tabletop[0, :2])
    assert not mapping.occupancy.occupied_mask()[cell]


@pytest.mark.parametrize('height,occupied', [(.15, True), (.63, True), (.64, False)])
def test_go2_navigation_collision_height_boundaries(height, occupied):
    g = grid(obstacle_height=(.15, .63))
    scan(g, [[1.05, .05, height]])
    cell = g.world_to_cell((1.05, .05))
    assert bool(g.occupied_mask()[cell]) is occupied


def test_ray_entirely_above_go2_collision_band_does_not_establish_ground_free_space():
    g = grid(obstacle_height=(.15, .63))
    scan(g, [[3.05, .05, .8]], origin=(.05, .05, .8))
    assert not g.observed.any()
    assert not g.occupied_mask().any()


def test_sloped_overhead_ray_only_observes_portion_below_collision_limit():
    g = grid(obstacle_height=(.15, .63))
    scan(g, [[3.05, .05, .8]], origin=(.05, .05, .4))
    near = g.world_to_cell((1.05, .05))  # Ray height is about .53 m.
    far = g.world_to_cell((2.05, .05))  # Ray height is about .67 m.
    assert g.observed[near] and not g.occupied_mask()[near]
    assert not g.observed[far]


def test_floor_rays_still_observe_empty_go2_route():
    g = grid(obstacle_height=(.15, .63))
    scan(g, [[3.05, .05, 0.]], origin=(.05, .05, .4))
    for xy in ((1.05, .05), (2.05, .05), (3.05, .05)):
        cell = g.world_to_cell(xy)
        assert g.observed[cell] and not g.occupied_mask()[cell]
        assert g.log_odds[cell] < 0


def test_clearance_at_one_measured_height_does_not_clear_another():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    scan(g, [[1.05, .05, .35], [1.05, .05, .85]], repeats=3)
    scan(g, [[3.05, .05, .35]], origin=(.05, .05, .35), repeats=60)
    assert g.occupied_mask()[cell]  # Upper obstacle is still supported.
    scan(g, [[3.05, .05, .85]], origin=(.05, .05, .85), repeats=60)
    assert g.observed[cell] and not g.occupied_mask()[cell]
    assert g.log_odds[cell] < 0


def test_same_frame_hit_wins_and_duplicate_returns_do_not_multiply_evidence():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    points = [[1.05, .05, .85]] * 100 + [[3.05, .05, 0.]]
    scan(g, points)
    assert g.log_odds[cell] == pytest.approx(.85)
    assert g.occupied_mask()[cell]


def test_hit_at_other_height_does_not_prevent_clearing_removed_lower_obstacle():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    scan(g, [[1.05, .05, .35], [1.05, .05, .85]], repeats=3)
    # Upper obstacle remains visible while lower-height rays pass through
    # the removed lower obstacle. They update independent height evidence.
    scan(g, [[1.05, .05, .85], [3.05, .05, .35]], origin=(.05, .05, .35), repeats=60)
    assert g.occupied_mask()[cell]
    scan(g, [[3.05, .05, .85]], origin=(.05, .05, .85), repeats=60)
    assert not g.occupied_mask()[cell]


def test_all_real_hits_survive_ray_budget_but_no_free_space_behind_near_hit():
    g = grid(max_lidar_rays=1)
    points = [[1.05, .05, .4], [3.05, .05, .4], [2.05, .85, .8]]
    scan(g, points)
    for point in points:
        assert g.occupied_mask()[g.world_to_cell(point[:2])]
    assert not g.observed[g.world_to_cell((2.05, .05))]


def test_footprint_clears_height_history_without_resurrection():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    scan(g, [[1.05, .05, .8]], repeats=3)
    g.mark_free((1.05, .05), .05)
    assert not g.occupied_mask()[cell]
    scan(g, [[3.05, .05, 0.]])
    assert g.log_odds[cell] < 0 and not g.occupied_mask()[cell]


def test_static_obstacle_survives_height_and_planar_clearing_and_footprint():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    g.mark_box_occupied((1.01, .01), (1.09, .09))
    scan(g, [[3.05, .05, .8]], origin=(.05, .05, .8), repeats=60)
    scan(g, [[3.05, .05, 0.]], repeats=60)
    g.mark_free((1.05, .05), .05)
    assert g.log_odds[cell] == 4 and g.occupied_mask()[cell]


def test_generic_obstacle_support_is_not_lost_by_height_projection():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    g.mark_occupied((1.05, .05), .05, evidence=2.55)
    scan(g, [[3.05, .05, 0.]])
    assert g.occupied_mask()[cell]
    assert g.log_odds[cell] == pytest.approx(2.475)


def test_pruning_and_external_point_updates_reset_and_preserve_height_state():
    g = grid()
    cell = g.world_to_cell((1.05, .05))
    g.update_obstacle_points(np.array([[1.05, .05, .8]] * 100))
    assert g.log_odds[cell] == pytest.approx(.85)
    g.prune_isolated_occupancy()
    scan(g, [[3.05, .05, 0.]])
    assert g.log_odds[cell] < 0 and not g.occupied_mask()[cell]
    g.update_obstacle_points(np.array([[1.05, .05, .8]]))
    assert g.occupied_mask()[cell]


@pytest.mark.parametrize('resolution', [0, -.1, float('nan'), float('inf')])
def test_invalid_height_resolution_is_rejected(resolution):
    with pytest.raises(ValueError, match='lidar_height_resolution'):
        grid(lidar_height_resolution=resolution)


def test_height_evidence_and_full_column_layers_are_not_silently_combined():
    with pytest.raises(ValueError, match='not both'):
        grid(lidar_free_height_layers=3)
