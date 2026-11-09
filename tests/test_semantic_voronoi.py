import unittest

import numpy as np

from grutopia_extension.interactive_navigation.mapping import (
    MappingConfig,
    OccupancyGridMap,
    PlanningError,
    SceneGraphNode,
    _bresenham,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import (
    SemanticVoronoiConfig,
    SemanticVoronoiGraph,
    VoronoiPathPlanner,
)


def _occupancy(rows=31, columns=41, resolution=0.1):
    config = MappingConfig(
        x_limits=(0.0, columns * resolution),
        y_limits=(0.0, rows * resolution),
        grid_resolution=resolution,
        robot_radius=resolution,
    )
    return OccupancyGridMap(config)


class SemanticVoronoiTest(unittest.TestCase):
    def test_empty_map_has_empty_graph(self):
        graph = SemanticVoronoiGraph(_occupancy())

        snapshot = graph.snapshot()

        self.assertEqual(snapshot.nodes, ())
        self.assertEqual(snapshot.edges, ())
        self.assertEqual(snapshot.frontiers, ())
        self.assertEqual(snapshot.skeleton_cells, ())

    def test_straight_corridor_compresses_to_endpoints_and_one_edge(self):
        occupancy = _occupancy()
        occupancy.observed[12:19, 3:38] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))

        snapshot = graph.snapshot()

        self.assertEqual(sum(node.kind == 'endpoint' for node in snapshot.nodes), 2)
        self.assertEqual(len(snapshot.edges), 1)
        self.assertGreater(snapshot.edges[0].length, 2.5)
        endpoint_cells = {node.cell for node in snapshot.nodes if node.kind == 'endpoint'}
        self.assertEqual({snapshot.edges[0].cells[0], snapshot.edges[0].cells[-1]}, endpoint_cells)
        self.assertTrue(all(cell[0] in range(14, 17) for cell in snapshot.skeleton_cells))

    def test_t_shape_produces_junction_and_three_corridors(self):
        occupancy = _occupancy()
        occupancy.observed[4:27, 18:23] = True
        occupancy.observed[4:9, 7:34] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))

        snapshot = graph.snapshot()

        self.assertGreaterEqual(sum(node.kind == 'junction' for node in snapshot.nodes), 1)
        self.assertEqual(sum(node.kind == 'endpoint' for node in snapshot.nodes), 3)
        self.assertEqual(len(snapshot.edges), 3)
        connected_kinds = {
            next(node.kind for node in snapshot.nodes if node.node_id == edge.source) for edge in snapshot.edges
        } | {next(node.kind for node in snapshot.nodes if node.node_id == edge.target) for edge in snapshot.edges}
        self.assertIn('junction', connected_kinds)

    def test_short_t_spur_is_pruned(self):
        occupancy = _occupancy()
        occupancy.observed[13:18, 3:38] = True
        occupancy.observed[9:14, 19:22] = True
        unpruned = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0)).snapshot()
        pruned = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.6)).snapshot()

        self.assertGreater(len(unpruned.edges), len(pruned.edges))
        self.assertEqual(len(pruned.edges), 1)

    def test_minimum_clearance_filters_unsafe_skeleton_cells(self):
        occupancy = _occupancy(rows=41, columns=41)
        occupancy.observed.fill(True)
        occupancy.log_odds[17:24, 17:24] = 4.0
        graph = SemanticVoronoiGraph(
            occupancy,
            SemanticVoronoiConfig(min_clearance=0.35, spur_length=0.0),
        )

        snapshot = graph.snapshot()

        self.assertTrue(snapshot.skeleton_cells)
        self.assertTrue(all(node.clearance >= 0.35 for node in snapshot.nodes))
        self.assertTrue(all(edge.min_clearance >= 0.35 for edge in snapshot.edges))
        self.assertTrue(all(not occupancy.inflated_mask()[cell] for cell in snapshot.skeleton_cells))

    def test_frontier_clusters_project_to_reachable_nodes(self):
        occupancy = _occupancy()
        occupancy.observed[8:23, 5:25] = True
        graph = SemanticVoronoiGraph(
            occupancy,
            SemanticVoronoiConfig(spur_length=0.0, frontier_min_size=2),
        )

        snapshot = graph.snapshot()

        self.assertEqual(len(snapshot.frontiers), 1)
        frontier = snapshot.frontiers[0]
        self.assertGreater(len(frontier.cells), 20)
        self.assertIsNotNone(frontier.node_id)
        self.assertIn(frontier.node_id, {node.node_id for node in snapshot.nodes})
        self.assertGreaterEqual(frontier.distance, 0.0)

    def test_semantic_objects_attach_to_nearest_voronoi_node(self):
        occupancy = _occupancy()
        occupancy.observed[12:19, 3:38] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        obj = SceneGraphNode(
            node_id='object:door:0',
            kind='object',
            position=(0.5, 1.5, 1.0),
            label='door',
        )

        attachments = graph.attach_semantics([obj])

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].object_id, obj.node_id)
        self.assertEqual(attachments[0].label, 'door')
        self.assertIsNotNone(attachments[0].node_id)
        self.assertLess(attachments[0].distance, 0.5)
        self.assertEqual(graph.snapshot().semantics, attachments)

    def test_semantic_object_outside_free_space_does_not_bind_across_components(self):
        occupancy = _occupancy()
        occupancy.observed[5:10, 3:15] = True
        occupancy.observed[20:25, 25:38] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        obj = SceneGraphNode(
            node_id='object:outside:0',
            kind='object',
            position=(-1.0, -1.0, 0.5),
            label='chair',
        )

        attachment = graph.attach_semantics([obj])[0]

        self.assertIsNone(attachment.node_id)
        self.assertIsNone(attachment.distance)

    def test_forced_update_drops_stale_skeleton_cells_after_new_inflation(self):
        occupancy = _occupancy()
        occupancy.observed.fill(True)
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        first = graph.snapshot()
        self.assertTrue(first.skeleton_cells)

        occupancy.log_odds[10:21, 10:21] = 4.0
        occupancy.revision += 1
        self.assertIs(graph.cached_snapshot(), first)

        rebuilt = graph.update(force=True)
        inflated = occupancy.inflated_mask()
        self.assertTrue(rebuilt.skeleton_cells)
        self.assertTrue(all(not inflated[cell] for cell in rebuilt.skeleton_cells))
        self.assertTrue(all(occupancy.observed[cell] for cell in rebuilt.skeleton_cells))

    def test_revision_cache_stable_ids_incremental_interface_and_serialization(self):
        occupancy = _occupancy()
        occupancy.observed[12:19, 3:38] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))

        first = graph.snapshot()
        cached = graph.snapshot()
        self.assertIs(first, cached)
        first_ids = {node.node_id for node in first.nodes}

        occupancy.revision += 1
        rebuilt = graph.update(changed_cells=[(15, 20)])

        self.assertIsNot(first, rebuilt)
        self.assertEqual(first_ids, {node.node_id for node in rebuilt.nodes})
        self.assertEqual(graph.statistics()['unchanged_updates'], 1)
        self.assertEqual(graph.statistics()['incremental_updates'], 0)
        self.assertGreaterEqual(graph.statistics()['cache_hits'], 1)
        payload = graph.to_dict()
        self.assertEqual(payload['revision'], occupancy.revision)
        self.assertEqual(len(payload['nodes']), len(rebuilt.nodes))

    def test_incremental_local_change_matches_full_rebuild(self):
        occupancy = _occupancy(rows=120, columns=160)
        occupancy.observed[20:27, 10:150] = True
        occupancy.observed[20:100, 30:37] = True
        occupancy.observed[20:100, 100:107] = True
        config = SemanticVoronoiConfig(spur_length=0.0, verify_incremental=True)
        graph = SemanticVoronoiGraph(occupancy, config)
        graph.snapshot()

        occupancy.observed[100:112, 30:37] = True
        occupancy.revision += 1
        incremental = graph.update()

        statistics = graph.statistics()
        self.assertEqual(statistics['builds'], 1)
        self.assertEqual(statistics['incremental_updates'], 1)
        self.assertEqual(statistics['incremental_fallbacks'], 0)
        self.assertEqual(statistics['incremental_verification_checks'], 1)
        self.assertEqual(statistics['incremental_verification_mismatches'], 0)
        self.assertGreater(statistics['incremental_seam_checks'], 0)
        self.assertGreater(statistics['last_window_cells'], 0)
        self.assertLess(statistics['last_window_cells'], occupancy.observed.size)

        reference = SemanticVoronoiGraph(occupancy, config).snapshot()
        self.assertEqual(
            {node.node_id for node in incremental.nodes},
            {node.node_id for node in reference.nodes},
        )
        self.assertEqual(
            {(edge.source, edge.target) for edge in incremental.edges},
            {(edge.source, edge.target) for edge in reference.edges},
        )
        self.assertEqual(
            {frontier.frontier_id for frontier in incremental.frontiers},
            {frontier.frontier_id for frontier in reference.frontiers},
        )
        self.assertEqual(set(incremental.skeleton_cells), set(reference.skeleton_cells))

    def test_full_verification_recovers_from_a_hidden_cache_mismatch(self):
        occupancy = _occupancy(rows=120, columns=160)
        occupancy.observed[20:27, 10:150] = True
        occupancy.observed[20:100, 30:37] = True
        occupancy.observed[20:100, 100:107] = True
        config = SemanticVoronoiConfig(spur_length=0.0, verify_incremental=True)
        graph = SemanticVoronoiGraph(occupancy, config)
        graph.snapshot()

        # Simulate a stale skeleton cell far from the next changed window.
        right_cells = np.argwhere(graph._last_skeleton[:, 110:])
        self.assertGreater(len(right_cells), 0)
        row, col = right_cells[0]
        graph._last_skeleton[int(row), int(col) + 110] = False

        occupancy.observed[100:112, 30:37] = True
        occupancy.revision += 1
        recovered = graph.update()
        statistics = graph.statistics()
        self.assertEqual(statistics['incremental_verification_mismatches'], 1)
        self.assertGreater(statistics['incremental_verification_mismatched_cells'], 0)
        self.assertEqual(statistics['last_incremental_fallback_reason'], 'verification_mismatch')
        self.assertEqual(statistics['incremental_fallbacks'], 1)
        reference = SemanticVoronoiGraph(occupancy, config).snapshot()
        self.assertEqual(set(recovered.skeleton_cells), set(reference.skeleton_cells))

    def test_incremental_distant_changes_use_separate_windows(self):
        occupancy = _occupancy(rows=120, columns=220)
        occupancy.observed[20:40, 10:60] = True
        occupancy.observed[80:100, 160:210] = True
        config = SemanticVoronoiConfig(spur_length=0.0)
        graph = SemanticVoronoiGraph(occupancy, config)
        graph.snapshot()

        # Two small additions are far enough apart that their guarded windows
        # do not overlap. A single bounding box would span almost the full map.
        occupancy.observed[35:45, 30:35] = True
        occupancy.observed[75:85, 180:185] = True
        occupancy.revision += 1
        incremental = graph.update()

        statistics = graph.statistics()
        self.assertEqual(statistics['last_changed_components'], 2)
        self.assertEqual(statistics['last_window_count'], 2)
        self.assertEqual(statistics['incremental_updates'], 1)
        self.assertEqual(statistics['incremental_fallbacks'], 0)
        self.assertEqual(statistics['incremental_fallback_reasons'], {})
        self.assertIsNone(statistics['last_incremental_fallback_reason'])
        self.assertLess(statistics['last_window_fraction'], config.incremental_max_window_fraction)
        self.assertGreaterEqual(statistics['last_window_work_cells'], statistics['last_window_cells'])
        self.assertGreater(statistics['last_window_to_changed_ratio'], 1.0)

        reference = SemanticVoronoiGraph(occupancy, config).snapshot()
        self.assertEqual(set(incremental.skeleton_cells), set(reference.skeleton_cells))

    def test_incremental_shortcut_reuses_graph_when_traversable_unchanged(self):
        occupancy = _occupancy(rows=120, columns=160)
        occupancy.observed[20:27, 10:150] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        first = graph.snapshot()

        occupancy.revision += 1
        refreshed = graph.update()

        self.assertEqual(graph.statistics()['builds'], 1)
        self.assertEqual(graph.statistics()['unchanged_updates'], 1)
        self.assertEqual(graph.statistics()['incremental_updates'], 0)
        self.assertEqual(graph.statistics()['last_window_count'], 0)
        self.assertEqual(graph.statistics()['last_window_cells'], 0)
        self.assertIsNone(graph.statistics()['last_incremental_fallback_reason'])
        self.assertEqual(refreshed.revision, occupancy.revision)
        self.assertEqual(first.nodes, refreshed.nodes)
        self.assertEqual(first.edges, refreshed.edges)

    def test_incremental_falls_back_to_full_rebuild_on_widespread_change(self):
        occupancy = _occupancy(rows=120, columns=160)
        occupancy.observed[20:27, 10:150] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        graph.snapshot()

        occupancy.observed[30:110, 10:150] = True
        occupancy.revision += 1
        rebuilt = graph.update()

        statistics = graph.statistics()
        self.assertEqual(statistics['builds'], 2)
        self.assertEqual(statistics['incremental_updates'], 0)
        self.assertEqual(statistics['incremental_fallbacks'], 1)
        fallback_reason = statistics['last_incremental_fallback_reason']
        self.assertIsNotNone(fallback_reason)
        self.assertEqual(statistics['incremental_fallback_reasons'][fallback_reason], 1)
        reference = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0)).snapshot()
        self.assertEqual(set(rebuilt.skeleton_cells), set(reference.skeleton_cells))

    def test_incremental_reach_uses_old_clearance_for_large_window_fallback(self):
        occupancy = _occupancy(rows=160, columns=180)
        occupancy.observed.fill(True)
        config = SemanticVoronoiConfig(spur_length=0.0, region_segmentation=False)
        graph = SemanticVoronoiGraph(occupancy, config)
        graph.snapshot()

        occupancy.log_odds[80, 90] = 4.0
        occupancy.revision += 1
        rebuilt = graph.update()

        statistics = graph.statistics()
        self.assertEqual(statistics['incremental_fallbacks'], 1)
        self.assertEqual(statistics['last_incremental_fallback_reason'], 'window_fraction')
        self.assertEqual(statistics['incremental_fallback_reasons'].get('reach_not_converged', 0), 0)
        reference = SemanticVoronoiGraph(occupancy, config).snapshot()
        self.assertEqual(set(rebuilt.skeleton_cells), set(reference.skeleton_cells))

    def test_two_rooms_connected_by_narrow_doorway_form_regions(self):
        occupancy = _occupancy(rows=60, columns=100)
        occupancy.observed[10:50, 10:40] = True
        occupancy.observed[10:50, 60:90] = True
        occupancy.observed[27:33, 40:60] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.3))

        snapshot = graph.snapshot()

        self.assertEqual(len(snapshot.regions), 2)
        self.assertEqual(len(snapshot.doorways), 1)
        doorway = snapshot.doorways[0]
        self.assertLessEqual(doorway.width, 0.8)
        self.assertEqual(set(doorway.regions), {region.region_id for region in snapshot.regions})
        for region in snapshot.regions:
            self.assertGreater(region.area, 5.0)
            self.assertEqual(len(region.adjacent), 1)
        region_ids = {region.region_id for region in snapshot.regions}
        for frontier in snapshot.frontiers:
            if frontier.node_id is not None:
                self.assertIn(frontier.region_id, region_ids)

    def test_semantic_labels_aggregate_into_owning_region(self):
        occupancy = _occupancy(rows=60, columns=100)
        occupancy.observed[10:50, 10:40] = True
        occupancy.observed[10:50, 60:90] = True
        occupancy.observed[27:33, 40:60] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.3))
        bed = SceneGraphNode(node_id='object:bed:0', kind='object', position=(2.0, 3.0, 0.5), label='bed')
        fridge = SceneGraphNode(
            node_id='object:fridge:0',
            kind='object',
            position=(7.5, 3.0, 0.5),
            label='refrigerator',
        )

        graph.attach_semantics([bed, fridge])
        snapshot = graph.cached_snapshot()

        labelled = {
            label: region.region_id
            for region in snapshot.regions
            for label, _count in region.labels
        }
        self.assertEqual(set(labelled), {'bed', 'refrigerator'})
        self.assertNotEqual(labelled['bed'], labelled['refrigerator'])

    def test_open_space_produces_single_region_without_doorways(self):
        occupancy = _occupancy(rows=41, columns=41)
        occupancy.observed.fill(True)
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))

        snapshot = graph.snapshot()

        self.assertEqual(len(snapshot.regions), 1)
        self.assertEqual(snapshot.doorways, ())
        self.assertEqual(
            set(snapshot.regions[0].node_ids),
            {node.node_id for node in snapshot.nodes},
        )


class VoronoiPathPlannerTest(unittest.TestCase):
    def test_l_corridor_path_follows_skeleton_and_avoids_obstacles(self):
        occupancy = _occupancy(rows=70, columns=45)
        occupancy.observed[12:19, 3:38] = True
        occupancy.observed[12:60, 31:38] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        planner = VoronoiPathPlanner(graph)

        path = planner.plan((0.5, 1.5, 0.0), (3.45, 5.5, 0.0))

        self.assertGreaterEqual(len(path), 3)
        self.assertAlmostEqual(path[-1][0], 3.45)
        self.assertAlmostEqual(path[-1][1], 5.5)
        skeleton_world = np.asarray(
            [occupancy.cell_to_world(cell) for cell in graph.snapshot().skeleton_cells]
        )
        for waypoint in path[:-1]:
            nearest = float(
                np.linalg.norm(skeleton_world - np.asarray(waypoint[:2]), axis=1).min()
            )
            self.assertLess(nearest, 0.40)
        blocked = occupancy.inflated_mask()
        previous = occupancy.world_to_cell((0.5, 1.5))
        for waypoint in path:
            cell = occupancy.world_to_cell(waypoint[:2])
            for ray_cell in _bresenham(previous, cell):
                self.assertFalse(blocked[ray_cell])
            previous = cell

    def test_disconnected_goal_raises_planning_error(self):
        occupancy = _occupancy(rows=40, columns=60)
        occupancy.observed[10:30, 5:25] = True
        occupancy.observed[10:30, 35:55] = True
        graph = SemanticVoronoiGraph(occupancy, SemanticVoronoiConfig(spur_length=0.0))
        planner = VoronoiPathPlanner(graph)

        with self.assertRaises(PlanningError):
            planner.plan((1.5, 2.0, 0.0), (4.5, 2.0, 0.0))

    def test_runtime_uses_voronoi_and_falls_back_to_grid_astar(self):
        from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime

        config = MappingConfig(
            x_limits=(0.0, 6.0),
            y_limits=(0.0, 4.0),
            grid_resolution=0.1,
            robot_radius=0.1,
        )
        runtime = MapNavigationRuntime(
            mapping_config=config,
            use_semantic_voronoi=True,
            prefer_voronoi_paths=True,
        )
        occupancy = runtime.map.occupancy
        occupancy.observed[12:19, 3:38] = True
        occupancy.revision += 1
        runtime.semantic_voronoi.update()

        path = runtime._map_plan((0.5, 1.5, 0.0), (3.5, 1.5, 0.0))
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(runtime.voronoi_plan_count, 1)
        self.assertEqual(runtime.voronoi_plan_fallbacks, 0)

        # Without any observations there is no skeleton, so the planner must
        # fall back to grid A*, which is allowed to cross unknown space.
        empty_runtime = MapNavigationRuntime(
            mapping_config=config,
            use_semantic_voronoi=True,
            prefer_voronoi_paths=True,
        )
        fallback_path = empty_runtime._map_plan((0.5, 1.5, 0.0), (5.5, 3.5, 0.0))
        self.assertGreaterEqual(len(fallback_path), 1)
        self.assertEqual(empty_runtime.voronoi_plan_count, 0)
        self.assertEqual(empty_runtime.voronoi_plan_fallbacks, 1)

    def test_static_obstacle_seeding_without_occupancy_marking(self):
        from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime

        config = MappingConfig(
            x_limits=(0.0, 6.0),
            y_limits=(0.0, 4.0),
            grid_resolution=0.1,
            robot_radius=0.1,
        )
        runtime = MapNavigationRuntime(mapping_config=config)
        runtime.seed_static_obstacles(
            [{'label': 'sofa/0', 'minimum_xy': (1.0, 1.0), 'maximum_xy': (2.0, 2.0)}],
            mark_occupancy=False,
        )

        self.assertEqual(len(runtime._static_obstacle_boxes), 1)
        self.assertFalse(runtime.map.occupancy.occupied_mask().any())
        self.assertFalse(runtime.map.occupancy.observed.any())


if __name__ == '__main__':
    unittest.main()
