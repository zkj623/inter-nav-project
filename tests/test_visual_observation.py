"""Visibility, evidence and cost regressions independent of Isaac Sim."""
import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, OccupancyGridMap
from grutopia_extension.interactive_navigation.visual_observation import VisualObservationMemory


def memory():
    occ = OccupancyGridMap(MappingConfig(x_limits=(-5., 5.), y_limits=(-5., 5.), grid_resolution=.1))
    occ.observed[:] = True
    return VisualObservationMemory(occ)


def camera(points, yaw=0., frame=1, origin=(0., 0., .7)):
    return dict(pointcloud=np.asarray(points), position=origin,
                orientation=(np.cos(yaw/2), 0., 0., np.sin(yaw/2)),
                horizontal_fov_degrees=96., rendering_frame=frame)


def test_only_actual_returns_count_and_fov_edges_floor_and_far_points_do_not():
    m = memory()
    m.observe(camera([[2., 0., .7], [2., 2., .7], [1., 0., .1], [4.5, 0., .7]]))
    occ = m.occupancy
    assert m.seen[occ.world_to_cell((2., 0.))]
    for xy in [(3., 0.), (2., 2.), (1., 0.), (4.5, 0.)]:
        assert not m.seen[occ.world_to_cell(xy)]


def test_opposite_surface_side_is_not_marked_and_cached_frame_is_ignored():
    m = memory()
    assert m.observe(camera([[2., 0., .7]]))
    before = m.seen.copy()
    assert not m.observe(camera([[-2., 0., .7]], yaw=np.pi))
    np.testing.assert_array_equal(before, m.seen)
    bits = m.seen[m.occupancy.world_to_cell((2., 0.))]
    assert bits & 1 and not bits & (1 << 4)


def test_geometry_without_camera_evidence_never_claims_visual_coverage():
    m = memory()
    m.occupancy.mark_box_occupied((-2., 1.), (2., 1.1))
    assert m.best_direction((0., 0.)) is None
    assert not m.seen.any()


def test_occluder_blocks_gain_from_far_wall():
    m = memory()
    m.observe(camera([[1., 0., .7]]))
    m.occupancy.mark_box_occupied((-4., 1.), (4., 1.1))
    near = m.best_direction((0., 0.))
    assert near is not None
    m.occupancy.mark_box_occupied((-4., 2.), (4., 2.1))
    far = m.best_direction((0., 0.))
    assert far == near


def test_unknown_space_blocks_speculative_gain_and_seen_surface_removes_stop():
    m = memory()
    m.observe(camera([[1., 0., .7]]))
    m.occupancy.mark_box_occupied((-2., 1.), (2., 1.1))
    assert m.best_direction((0., 0.)) is not None
    m.seen[:] = 255
    assert m.best_direction((0., 0.)) is None
    m.seen[:] = 0
    m.occupancy.observed[:] = False
    assert m.best_direction((0., 0.)) is None


def test_shorter_turn_wins_for_symmetric_surface_gain():
    m = memory()
    m.observe(camera([[.3, 0., .7]]))
    m.occupancy.mark_box_occupied((-3., 1.5), (3., 1.6))
    look = m.best_direction((0., 0.))
    assert look is not None and 0 < look.yaw <= np.pi / 2
    assert look.gain >= .3
