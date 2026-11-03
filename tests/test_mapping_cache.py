"""Preserve navigation results when reusing inflation or moving voxel work."""
from dataclasses import replace

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.mapping import (
    FusedMap, MappingConfig, OccupancyGridMap, SemanticDetection,
)


def grid():
    return OccupancyGridMap(MappingConfig(
        x_limits=(0., 3.), y_limits=(0., 3.), grid_resolution=.1,
        obstacle_inflation_radius=.2, static_obstacle_inflation_radius=.4))


def test_inflation_reuses_unchanged_masks_across_evidence_updates(monkeypatch):
    occupancy = grid()
    calls = []
    original = occupancy._inflate
    def count(mask, radius):
        calls.append(radius)
        return original(mask, radius)
    monkeypatch.setattr(occupancy, '_inflate', count)
    occupancy.mark_occupied((1.05, 1.05), radius=0.)
    first = occupancy.inflated_mask()
    occupancy.mark_occupied((1.05, 1.05), radius=0.)
    # Evidence and revision changed; the effective obstacle masks did not.
    np.testing.assert_array_equal(occupancy.inflated_mask(), first)
    assert len(calls) == 2
    occupancy.mark_free((1.05, 1.05), radius=.1)
    assert not occupancy.inflated_mask().any()
    assert len(calls) == 4


def test_inflation_detects_direct_public_array_edits_without_revision_change():
    occupancy = grid()
    center = (15, 15)
    occupancy.log_odds[center] = .85
    dynamic = occupancy.inflated_mask()
    revision = occupancy.revision
    assert dynamic[15, 17] and not dynamic[15, 18]
    # Same occupied center becomes static and must use its larger radius.
    occupancy.static_occupied[center] = True
    static = occupancy.inflated_mask()
    assert static[15, 18]
    assert static.sum() > dynamic.sum()
    occupancy.static_occupied[center] = False
    occupancy.log_odds[center] = 0.
    assert not occupancy.inflated_mask().any()
    assert occupancy.revision == revision


def test_inflation_result_is_independent_of_cache_and_other_callers():
    occupancy = grid()
    occupancy.log_odds[15, 15] = .85
    first = occupancy.inflated_mask()
    expected = first.copy()
    first.fill(False)
    second = occupancy.inflated_mask()
    np.testing.assert_array_equal(second, expected)
    second.fill(True)
    np.testing.assert_array_equal(occupancy.inflated_mask(), expected)


@pytest.mark.parametrize('change', [
    {'obstacle_inflation_radius': .4},
    {'static_obstacle_inflation_radius': .6},
    {'grid_resolution': .05},
    {'occupied_threshold': 1.5},
])
def test_inflation_detects_replaced_configuration(change):
    occupancy = grid()
    occupancy.log_odds[7, 7] = .85
    occupancy.static_occupied[22, 22] = True
    previous = occupancy.inflated_mask()
    occupancy.config = replace(occupancy.config, **change)
    # A new object with the same public evidence is an uncached reference.
    reference = OccupancyGridMap(occupancy.config)
    reference.log_odds = occupancy.log_odds.copy()
    reference.static_occupied = occupancy.static_occupied.copy()
    actual = occupancy.inflated_mask()
    np.testing.assert_array_equal(actual, reference.inflated_mask())
    assert not np.array_equal(actual, previous)


def assert_same_navigation(left, right):
    for name in ('log_odds', 'observed', 'static_occupied', '_height_evidence', '_column_evidence'):
        np.testing.assert_array_equal(getattr(left.occupancy, name), getattr(right.occupancy, name))
    np.testing.assert_array_equal(left.occupancy.inflated_mask(), right.occupancy.inflated_mask())
    assert left.lidar_frames == right.lidar_frames
    assert left.lidar_points == right.lidar_points
    assert left.rgb_frames == right.rgb_frames
    assert left.rgb_points == right.rgb_points
    assert left.scene_graph.object_nodes() == right.scene_graph.object_nodes()


def test_disabling_voxels_preserves_lidar_navigation_and_counters(monkeypatch):
    config = MappingConfig(x_limits=(0., 4.), y_limits=(-1., 1.),
                           obstacle_height=(.15, .63), lidar_height_resolution=.1,
                           voxel_clear_observations=3, sticky_free_scale=.25)
    original = FusedMap(config)
    navigation = FusedMap(config, maintain_voxels=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Display voxel maintenance ran in navigation')
    monkeypatch.setattr(navigation.voxels, 'update', forbidden)
    monkeypatch.setattr(navigation.voxels, 'clear_lidar_free_space', forbidden)
    scans = [np.array([[1.05, .05, .35], [1.05, .05, .9], [2.05, -.45, 0.]]),
             np.array([[3.05, .05, .35], [3.05, .25, 0.]])]
    for step, scan in enumerate([scans[0]]*3 + [scans[1]]*5):
        for mapping in (original, navigation):
            mapping.update_lidar((.05, .05, .35), scan, step, 8.)
        assert_same_navigation(original, navigation)
    assert original.voxels.voxel_count > 0
    assert navigation.voxels.voxel_count == 0


@pytest.mark.parametrize('clear_free_space', [False, True])
def test_disabling_voxels_keeps_rgb_geometry_and_semantic_detections(clear_free_space):
    config = MappingConfig(x_limits=(0., 4.), y_limits=(-1., 1.),
                           obstacle_height=(.15, .63), lidar_height_resolution=.1)
    original = FusedMap(config)
    navigation = FusedMap(config, maintain_voxels=False)
    points = np.array([[1.05, .05, .35], [2.05, .25, .45]])
    colors = np.array([[100, 150, 200], [50, 60, 70]])
    detections = [SemanticDetection('obstacle', (1.05, .05, .35), step=1)]
    for mapping in (original, navigation):
        mapping.update_rgb(points, colors, detections, step=1,
                           origin=(.05, .05, .35), clear_free_space=clear_free_space)
    assert_same_navigation(original, navigation)
    assert original.voxels.colored_voxel_count == 2
    assert navigation.voxels.voxel_count == 0
