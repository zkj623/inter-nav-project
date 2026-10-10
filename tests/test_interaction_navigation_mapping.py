import unittest

import numpy as np

from grutopia_extension.interactive_navigation.mapping import (
    AStarMapPlanner,
    FusedMap,
    MappingConfig,
    OccupancyGridMap,
    SceneGraphMap,
    SemanticDetection,
    VoxelPointCloudMap,
)
from grutopia_extension.interactive_navigation.mapping_runtime import (
    MapNavigationRuntime,
    SemanticDetectionMode,
    _camera_cloud,
    _deduplicate_semantic_detections,
    _semantic_detections,
)
from grutopia_extension.interactive_navigation.state_machine import (
    ControllerCommand,
    InteractionState,
    StateMachineDecision,
)


class InteractionNavigationMappingTest(unittest.TestCase):
    def test_ray_budget_preserves_nearest_thin_obstacle_before_flattening(self):
        config = MappingConfig(
            x_limits=(0.0, 8.0), y_limits=(-2.0, 2.0),
            grid_resolution=0.1, robot_radius=0.1, max_lidar_rays=2,
        )
        # An input-point sampler retaining only first/last returns misses the
        # leg. Reversing the frame must not change the resulting occupancy.
        frame = np.array([[6.5, 0.0, 0.5], [2.5, 0.0, 0.5], [6.5, 0.0, 0.0]])
        maps = []
        for points in (frame, frame[::-1]):
            occupancy = OccupancyGridMap(config)
            occupancy.update_lidar((0.5, 0.0, 0.5), points, max_range=8.0)
            self.assertTrue(occupancy.occupied_mask()[occupancy.world_to_cell((2.5, 0.0))])
            self.assertFalse(occupancy.observed[occupancy.world_to_cell((4.5, 0.0))])
            maps.append(occupancy.log_odds)
        np.testing.assert_array_equal(*maps)


    def test_offset_lidar_does_not_clear_a_robot_footprint_around_sensor(self):
        occupancy = OccupancyGridMap(MappingConfig(
            x_limits=(0.0, 3.0), y_limits=(-1.0, 1.0), robot_radius=0.3,
        ))
        # Sensor is 24 cm ahead of the base. A measured obstacle 26 cm ahead
        # of the sensor is outside the base footprint and must remain blocked.
        occupancy.update_lidar((0.75, 0.0, 0.5), np.array([[1.01, 0.0, 0.5]]), max_range=8.0)
        occupancy.mark_free((0.51, 0.0), radius=0.3)
        self.assertTrue(occupancy.occupied_mask()[occupancy.world_to_cell((1.01, 0.0))])


    def test_same_frame_hit_wins_over_crossing_free_ray(self):
        config = MappingConfig(x_limits=(0, 6), y_limits=(-3, 3), sticky_free_scale=0.25)
        points = np.array([[0.35, 0.04, 0.5], [5, 0.1, 0]])
        for frame in (points, points[::-1]):
            occupancy = OccupancyGridMap(config)
            occupancy.update_lidar((0.05, 0.05, 0.5), frame, max_range=8)
            cell = occupancy.world_to_cell((0.35, 0.04))
            self.assertAlmostEqual(occupancy.log_odds[cell], config.occupied_update, places=5)


    def test_lidar_frame_density_does_not_override_history(self):
        config = MappingConfig(x_limits=(0, 6), y_limits=(-3, 3), sticky_free_scale=0.25)
        origin = (0.05, 0.05, 0.5)
        angles = np.linspace(-0.16, 0.16, 80)
        for radius, height, initial, expected in [(5, 0, 4, 3.925), (0.3, 0.5, -4, -3.15)]:
            with self.subTest(height=height):
                occupancy = OccupancyGridMap(config)
                cell = occupancy.world_to_cell((0.35, 0.05))
                occupancy.log_odds[cell] = initial
                occupancy.observed[cell] = True
                points = np.column_stack((
                    origin[0] + radius * np.cos(angles),
                    origin[1] + radius * np.sin(angles), np.full(len(angles), height),
                ))
                occupancy.update_lidar(origin, points, max_range=8)
                self.assertAlmostEqual(occupancy.log_odds[cell], expected, places=5)
                # New independent frames can still correct historical evidence.
                for _ in range(60):
                    occupancy.update_lidar(origin, points, max_range=8)
                self.assertEqual(bool(occupancy.occupied_mask()[cell]), height > 0)


    def test_same_frame_semantic_sources_are_deduplicated(self):
        detections = _deduplicate_semantic_detections(
            (
                SemanticDetection(
                    'chair',
                    (1.0, 1.0, 0.5),
                    confidence=0.8,
                    embedding=(1.0, 0.0),
                    point_count=20,
                    step=7,
                    sources=('open_vocabulary',),
                ),
                SemanticDetection(
                    'chair',
                    (1.1, 1.0, 0.5),
                    confidence=1.0,
                    color=(10, 20, 30),
                    step=7,
                    sources=('isaac',),
                ),
            )
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].position, (1.0, 1.0, 0.5))
        self.assertEqual(detections[0].embedding, (1.0, 0.0))
        self.assertEqual(detections[0].color, (10, 20, 30))
        self.assertEqual(detections[0].sources, ('open_vocabulary', 'isaac'))
        graph = SceneGraphMap(MappingConfig())
        graph.update_detections(detections)
        self.assertEqual(graph.object_nodes()[0].observations, 1)
        self.assertEqual(graph.object_nodes()[0].sources, ('open_vocabulary', 'isaac'))

    def test_same_frame_door_merge_keeps_refrigerator_label_evidence(self):
        graph = SceneGraphMap(MappingConfig())
        for step in range(8):
            door = SemanticDetection(
                'door', (2.0, 0.0, 1.0), confidence=0.55,
                embedding=(1.0, 0.0), step=step,
            )
            refrigerator = SemanticDetection(
                'refrigerator door', (2.1, 0.0, 1.0), confidence=0.40,
                embedding=(1.0, 0.0), step=step,
            )
            detections = _deduplicate_semantic_detections(
                (door, refrigerator) if step % 2 == 0 else (refrigerator, door)
            )
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0].label, 'door')
            self.assertEqual(set(detections[0].label_evidence), {'door', 'refrigerator door'})
            graph.update_detections(detections)

        node = graph.object_nodes()[0]
        self.assertEqual(node.observations, 8)
        self.assertEqual(graph.label_counts(node.node_id), {'door': 8, 'refrigerator door': 8})

    def test_voxel_map_fuses_lidar_geometry_and_rgb_color(self):
        config = MappingConfig(voxel_size=0.2)
        voxel_map = VoxelPointCloudMap(config)
        lidar_points = np.array([[1.01, 0.01, 0.5], [1.08, 0.04, 0.52]])
        rgb_points = np.array([[1.05, 0.02, 0.51], [2.0, 0.0, 0.5]])
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        voxel_map.update(lidar_points, step=1, source='lidar')
        voxel_map.update(rgb_points, colors=colors, step=2, source='rgb')

        self.assertEqual(voxel_map.voxel_count, 2)
        self.assertEqual(voxel_map.lidar_voxel_count, 1)
        self.assertEqual(voxel_map.colored_voxel_count, 2)
        self.assertEqual(voxel_map.points().shape, (2, 3))

    def test_lidar_rays_mark_free_space_and_obstacle_endpoints(self):
        config = MappingConfig(
            x_limits=(0.0, 5.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        origin = (0.5, 0.0, 1.0)
        points = np.array([[3.0, 0.0, 0.8], [3.0, 1.0, 0.0]])

        occupancy.update_lidar(origin, points, max_range=5.0)

        obstacle_cell = occupancy.world_to_cell((3.0, 0.0))
        free_cell = occupancy.world_to_cell((1.5, 0.0))
        floor_endpoint = occupancy.world_to_cell((3.0, 1.0))
        self.assertTrue(occupancy.occupied_mask()[obstacle_cell])
        self.assertFalse(occupancy.occupied_mask()[free_cell])
        self.assertFalse(occupancy.occupied_mask()[floor_endpoint])
        self.assertTrue(occupancy.observed[free_cell])

    def test_flattened_scan_blocks_free_space_behind_nearer_obstacle(self):
        """A tabletop return in the same direction as a far wall must clamp
        the free-space ray, leaving the space behind the table unknown."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        origin = (0.5, 0.0, 0.45)
        frame = np.array([[2.5, 0.0, 0.75], [6.5, 0.0, 0.45]])

        occupancy.update_lidar(origin, frame, max_range=8.0)

        table_edge = occupancy.world_to_cell((2.5, 0.0))
        in_front = occupancy.world_to_cell((1.5, 0.0))
        behind_table = occupancy.world_to_cell((4.5, 0.0))
        self.assertTrue(occupancy.occupied_mask()[table_edge])
        self.assertTrue(occupancy.observed[in_front])
        self.assertFalse(occupancy.occupied_mask()[in_front])
        self.assertFalse(occupancy.observed[behind_table])

    def test_couch_seat_confirmed_by_repeat_hits_survives_boundary_jitter(self):
        """A seat whose returns jitter around the band boundary is confirmed
        after a few in-band hits and must then survive frames that only see
        the below-band skirt and the floor behind it."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.10,
            robot_radius=0.1,
            obstacle_height=(0.20, 1.6),
        )
        occupancy = OccupancyGridMap(config)
        origin = (5.0, 0.0, 0.45)
        seat = np.array([[2.0, 0.0, 0.34]])
        for _ in range(2):
            occupancy.update_lidar(origin, seat, max_range=8.0)
        seat_cell = occupancy.world_to_cell((2.0, 0.0))
        self.assertGreaterEqual(occupancy.log_odds[seat_cell], config.sticky_occupied_threshold)

        skirt_and_floor = np.array([[2.15, 0.0, 0.17], [0.4, 0.0, 0.05]])
        for _ in range(60):
            occupancy.update_lidar(origin, skirt_and_floor, max_range=8.0)

        self.assertTrue(
            occupancy.occupied_mask()[seat_cell],
            'confirmed seat must survive frames without in-band returns',
        )
        front_cell = occupancy.world_to_cell((3.5, 0.0))
        self.assertTrue(occupancy.observed[front_cell])
        self.assertFalse(occupancy.occupied_mask()[front_cell])

    def test_prune_resets_isolated_weak_occupancy_only(self):
        """Lone unconfirmed cells (floor noise) reset to unknown; contiguous
        furniture and confirmed thin obstacles must survive pruning."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.10,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        noise_cell = occupancy.world_to_cell((3.0, 1.0))
        occupancy.log_odds[noise_cell] = 0.85  # single unconfirmed hit
        pole_cell = occupancy.world_to_cell((5.0, -1.0))
        occupancy.log_odds[pole_cell] = 2.5  # confirmed thin obstacle
        wall_cells = [occupancy.world_to_cell((6.0, y)) for y in (0.0, 0.1, 0.2)]
        for cell in wall_cells:
            occupancy.log_odds[cell] = 0.85  # contiguous unconfirmed wall

        occupancy.prune_isolated_occupancy()

        self.assertLess(occupancy.log_odds[noise_cell], config.occupied_threshold)
        self.assertTrue(occupancy.occupied_mask()[pole_cell])
        for cell in wall_cells:
            self.assertTrue(occupancy.occupied_mask()[cell])

    def test_confirmed_occupancy_is_immune_to_free_rays(self):
        """With sticky scale zero, confirmed obstacles never fade under free
        rays; only physically traversing the cell clears them."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.10,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        couch_cell = occupancy.world_to_cell((2.0, 0.0))
        occupancy.log_odds[couch_cell] = 2.0

        origin = (5.0, 0.0, 0.45)
        floor_behind = np.array([[0.4, 0.0, 0.05]])
        for _ in range(200):
            occupancy.update_lidar(origin, floor_behind, max_range=8.0)
        self.assertEqual(float(occupancy.log_odds[couch_cell]), 2.0)

        occupancy.mark_free((2.0, 0.0), radius=0.15)
        self.assertFalse(occupancy.occupied_mask()[couch_cell])

    def test_isolated_floor_noise_does_not_block_free_space(self):
        """A guard-band return far from any occupancy (floor clutter) must
        not clamp free rays, otherwise corridors would never be confirmed."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.10,
            robot_radius=0.1,
            obstacle_height=(0.28, 1.6),
        )
        occupancy = OccupancyGridMap(config)
        origin = (5.0, 0.0, 0.45)
        # Isolated near-floor return at 3.0 m plus the floor far behind it.
        noise_and_floor = np.array([[3.0, 0.0, 0.20], [0.5, 0.0, 0.05]])
        occupancy.update_lidar(origin, noise_and_floor, max_range=8.0)

        behind_noise = occupancy.world_to_cell((1.5, 0.0))
        self.assertTrue(
            occupancy.observed[behind_noise],
            'free rays must pass isolated floor noise and confirm the corridor',
        )
        self.assertFalse(occupancy.occupied_mask()[behind_noise])

    def test_unconfirmed_obstacles_fade_under_free_rays(self):
        """A single-hit (unconfirmed) obstacle may still fade when later rays
        keep reporting the direction as clear."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        stool = np.array([[2.0, 0.0, 0.35]])
        occupancy.update_lidar((6.0, 0.0, 0.45), stool, max_range=8.0)
        stool_cell = occupancy.world_to_cell((2.0, 0.0))
        self.assertTrue(occupancy.occupied_mask()[stool_cell])

        near_origin = (2.5, 0.0, 0.45)
        floor_behind = np.array([[0.5, 0.0, 0.05]])
        for _ in range(80):
            occupancy.update_lidar(near_origin, floor_behind, max_range=8.0)
        self.assertFalse(occupancy.occupied_mask()[stool_cell])

    def test_confirmed_furniture_survives_underpassing_lidar_rays(self):
        """Tabletop cells confirmed by camera points must not be washed away
        by lidar rays that pass underneath and hit the wall behind."""
        config = MappingConfig(
            x_limits=(0.0, 8.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        table_cell = occupancy.world_to_cell((3.0, 0.0))
        # Camera sees the tabletop: strong accumulated occupancy evidence.
        tabletop = np.array([[3.0, 0.0, 0.75]] * 4)
        occupancy.update_obstacle_points(tabletop)
        self.assertTrue(occupancy.occupied_mask()[table_cell])

        # Many lidar sweeps pass under the table and hit the far wall.
        origin = (0.5, 0.0, 0.45)
        wall_hits = np.array([[7.0, 0.0, 0.45]])
        for _ in range(30):
            occupancy.update_lidar(origin, wall_hits, max_range=8.0)

        self.assertTrue(
            occupancy.occupied_mask()[table_cell],
            'confirmed furniture must stay occupied under free-ray erosion',
        )

        # A weakly-evidenced cell (single hit) is still allowed to fade.
        transient = occupancy.world_to_cell((5.0, 0.0))
        occupancy.log_odds[transient] = config.occupied_threshold + 0.1
        for _ in range(10):
            occupancy.update_lidar(origin, wall_hits, max_range=8.0)
        self.assertFalse(occupancy.occupied_mask()[transient])

    def test_independent_safety_inflation_expands_obstacles(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.1,
            robot_radius=0.1,
            obstacle_inflation_radius=0.5,
        )
        occupancy = OccupancyGridMap(config)
        center = occupancy.world_to_cell((2.0, 0.0))
        occupancy.log_odds[center] = 2.0
        expanded = occupancy.world_to_cell((2.4, 0.0))

        self.assertFalse(occupancy.occupied_mask()[expanded])
        self.assertTrue(occupancy.inflated_mask()[expanded])

    def test_astar_routes_through_mapped_wall_gap(self):
        config = MappingConfig(
            x_limits=(0.0, 5.0),
            y_limits=(0.0, 5.0),
            grid_resolution=0.5,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        occupancy.observed.fill(True)
        wall_col = 5
        occupancy.log_odds[:, wall_col] = 2.0
        occupancy.log_odds[7, wall_col] = -2.0
        planner = AStarMapPlanner(occupancy)

        path = planner.plan((0.5, 0.5, 1.05), (4.5, 0.5, 1.05))

        self.assertEqual(path[-1], (4.5, 0.5, 1.05))
        self.assertTrue(any(point[1] >= 3.5 for point in path))
        for point in path[:-1]:
            cell = occupancy.world_to_cell(point[:2])
            self.assertFalse(occupancy.inflated_mask()[cell])

    def test_scene_graph_contains_places_geometry_and_semantics(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(0.0, 4.0),
            grid_resolution=0.25,
            graph_stride=2,
            robot_radius=0.1,
        )
        occupancy = OccupancyGridMap(config)
        occupancy.observed.fill(True)
        occupancy.log_odds[6:9, 6:9] = 2.0
        voxels = VoxelPointCloudMap(config)
        voxels.update(
            np.array([[1.7, 1.7, 0.5], [1.8, 1.8, 0.6]]),
            colors=np.array([[230, 20, 20], [240, 30, 20]]),
            step=1,
            source='rgb',
        )
        graph_map = SceneGraphMap(config)
        graph_map.update_detections(
            [SemanticDetection(label='door', position=(3.0, 2.0, 1.0), color=(120, 80, 30))]
        )

        graph = graph_map.snapshot(occupancy, voxels)

        self.assertGreater(graph.place_count, 0)
        self.assertGreaterEqual(graph.object_count, 2)
        self.assertTrue(any(node.label == 'door' for node in graph.nodes))
        self.assertTrue(any(node.label == 'red_geometry' for node in graph.nodes))
        self.assertTrue(any(edge.relation == 'near' for edge in graph.edges))

    def test_scene_graph_merges_cross_label_embedding_evidence(self):
        graph_map = SceneGraphMap(MappingConfig())
        graph_map.update_detections(
            [
                SemanticDetection(
                    label='sofa',
                    position=(1.0, 0.0, 0.5),
                    confidence=0.7,
                    embedding=(1.0, 0.0),
                    point_count=20,
                    step=2,
                ),
                SemanticDetection(
                    label='couch',
                    position=(1.1, 0.0, 0.5),
                    confidence=0.75,
                    embedding=(0.99, 0.01),
                    point_count=30,
                    step=4,
                ),
            ]
        )

        nodes = graph_map.object_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].observations, 2)
        self.assertEqual(nodes[0].point_count, 50)
        self.assertEqual(nodes[0].last_seen_step, 4)
        self.assertEqual(graph_map.label_counts(nodes[0].node_id), {'sofa': 1, 'couch': 1})
        self.assertAlmostEqual(np.linalg.norm(nodes[0].embedding), 1.0)

    def test_open_door_semantic_node_becomes_traversable(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.1,
            robot_radius=0.1,
        )
        fused_map = FusedMap(config)
        detections = [
            SemanticDetection('door', (2.0, 0.0, 1.0)),
            SemanticDetection('door', (2.0, 0.0, 1.0)),
        ]
        fused_map.scene_graph.update_detections(detections)
        fused_map._reinforce_semantic_occupancy()
        door_cell = fused_map.occupancy.world_to_cell((2.0, 0.0))
        self.assertTrue(fused_map.occupancy.occupied_mask()[door_cell])

        fused_map.set_semantic_labels_traversable({'door'})

        self.assertFalse(fused_map.occupancy.occupied_mask()[door_cell])
        self.assertIn('door', fused_map.statistics()['traversable_semantic_labels'])

    def test_fused_map_reports_mapping_and_planning_statistics(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        fused_map = FusedMap(config)
        lidar_points = np.array([[3.0, -1.0, 0.5], [3.0, 1.0, 0.5], [3.5, 0.0, 0.0]])
        fused_map.update_lidar((0.5, 0.0, 1.0), lidar_points, step=1, max_range=4.0)
        fused_map.update_rgb(
            np.array([[2.0, 0.5, 0.5]]),
            np.array([[10, 20, 240]], dtype=np.uint8),
            [SemanticDetection('obstacle', (2.0, 0.5, 0.5))],
            step=1,
        )
        path = fused_map.plan((0.5, 0.0, 1.05), (2.5, 0.0, 1.05))
        stats = fused_map.statistics()

        self.assertGreater(len(path), 0)
        self.assertEqual(stats['lidar_frames'], 1)
        self.assertEqual(stats['rgb_frames'], 1)
        self.assertGreater(stats['voxel_count'], 0)
        self.assertGreater(stats['colored_voxel_count'], 0)
        self.assertEqual(stats['plans'], 1)
        semantic_cell = fused_map.occupancy.world_to_cell((2.0, 0.5))
        self.assertTrue(fused_map.occupancy.occupied_mask()[semantic_cell])

    def test_camera_frame_aligns_rgb_with_world_points_and_semantics(self):
        camera = {
            'rgba': np.array(
                [
                    [[255, 0, 0, 255], [250, 10, 0, 255]],
                    [[0, 0, 0, 255], [240, 20, 0, 255]],
                ],
                dtype=np.uint8,
            ),
            'depth': np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            'pointcloud': np.array(
                [[1.0, 0.0, 0.5], [1.0, 0.1, 0.5], [1.0, 0.2, 0.5]],
                dtype=np.float32,
            ),
            'bounding_box_2d_tight': {
                'data': np.array([[7, 0, 0, 2, 2, 0.0]], dtype=np.float32),
                'info': {'idToLabels': {'7': {'class': 'obstacle'}}},
            },
        }

        points, colors, point_image = _camera_cloud(camera)
        detections = _semantic_detections(camera, point_image, step=17)

        self.assertEqual(points.shape, (3, 3))
        self.assertEqual(colors.shape, (3, 3))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, 'obstacle')
        self.assertGreater(detections[0].color[0], 200)
        self.assertEqual(detections[0].step, 17)
        self.assertEqual(detections[0].sources, ('isaac',))

    def test_scene_graph_ignores_floor_and_robot_semantic_boxes(self):
        camera = {
            'rgba': np.full((2, 2, 4), 255, dtype=np.uint8),
            'bounding_box_2d_tight': {
                'data': np.array(
                    [
                        [1, 0, 0, 2, 2, 0.0],
                        [2, 0, 0, 2, 2, 0.0],
                    ],
                    dtype=np.float32,
                ),
                'info': {
                    'idToLabels': {
                        '1': {'class': 'floor'},
                        '2': {'class': 'g1'},
                    }
                },
            },
        }
        point_image = np.ones((2, 2, 3), dtype=np.float32)

        self.assertEqual(_semantic_detections(camera, point_image), [])

    def test_camera_cloud_filters_full_image_pointcloud_with_depth_mask(self):
        camera = {
            'rgba': np.full((2, 2, 4), 255, dtype=np.uint8),
            'depth': np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            'pointcloud': np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [99.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        }

        points, _, point_image = _camera_cloud(camera)

        np.testing.assert_allclose(points[:, 0], [1.0, 2.0, 4.0])
        self.assertTrue(np.isnan(point_image[1, 0]).all())
        self.assertEqual(point_image[1, 1, 0], 4.0)

    def test_rgb_can_color_voxels_without_changing_lidar_occupancy(self):
        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
        )
        camera = {
            'rgba': np.full((1, 1, 4), 200, dtype=np.uint8),
            'depth': np.ones((1, 1), dtype=np.float32),
            'pointcloud': np.array([[2.0, 0.0, 0.5]], dtype=np.float32),
            'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
            'bounding_box_2d_tight': {
                'data': np.array([[1, 0, 0, 1, 1, 0.0]], dtype=np.float32),
                'info': {'idToLabels': {'1': {'class': 'obstacle'}}},
            },
        }

        runtime.update(
            0,
            {
                'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
                'sensors': {'camera': camera},
            },
        )

        self.assertEqual(runtime.map.rgb_frames, 1)
        self.assertEqual(runtime.map.voxels.colored_voxel_count, 1)
        self.assertFalse(runtime.map.occupancy.occupied_mask().any())
        self.assertEqual(len(runtime.map.scene_graph.object_nodes()), 1)

    def test_open_vocabulary_detections_merge_with_camera_semantics(self):
        class _FakePerception:
            def perceive(self, **kwargs):
                return [SemanticDetection('refrigerator', (2.0, 0.0, 0.5))]

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_FakePerception(),
        )
        camera = {
            'rgba': np.full((1, 1, 4), 200, dtype=np.uint8),
            'depth': np.ones((1, 1), dtype=np.float32),
            'pointcloud': np.array([[2.0, 0.0, 0.5]], dtype=np.float32),
            'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
            'bounding_box_2d_tight': {
                'data': np.array([[1, 0, 0, 1, 1, 0.0]], dtype=np.float32),
                'info': {'idToLabels': {'1': {'class': 'obstacle'}}},
            },
        }

        runtime.update(
            0,
            {
                'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
                'sensors': {'camera': camera},
            },
        )

        labels = sorted(node.label for node in runtime.map.scene_graph.object_nodes())
        self.assertEqual(labels, ['obstacle', 'refrigerator'])
        refrigerator = next(
            node for node in runtime.map.scene_graph.object_nodes() if node.label == 'refrigerator'
        )
        self.assertEqual(refrigerator.sources, ('open_vocabulary',))
        self.assertEqual(runtime.open_vocabulary_attempts, 1)
        self.assertEqual(runtime.open_vocabulary_frames, 1)
        self.assertEqual(runtime.open_vocabulary_detections, 1)
        self.assertEqual(runtime.open_vocabulary_last_labels, ['refrigerator'])

    def test_detection_mode_accepts_cli_spellings(self):
        self.assertIs(SemanticDetectionMode.parse('isaac'), SemanticDetectionMode.ISAAC)
        self.assertIs(
            SemanticDetectionMode.parse('open-vocabulary'),
            SemanticDetectionMode.OPEN_VOCABULARY,
        )
        self.assertIs(SemanticDetectionMode.parse('HYBRID'), SemanticDetectionMode.HYBRID)
        self.assertIs(
            SemanticDetectionMode.parse(SemanticDetectionMode.HYBRID),
            SemanticDetectionMode.HYBRID,
        )
        with self.assertRaises(ValueError):
            SemanticDetectionMode.parse('grounding-dino')

    def test_open_vocabulary_mode_without_perception_is_rejected(self):
        with self.assertRaises(ValueError):
            MapNavigationRuntime(
                mapping_config=MappingConfig(),
                semantic_detection_mode='open_vocab',
            )

    def test_isaac_mode_skips_open_vocabulary_perception(self):
        class _FakePerception:
            calls = 0

            def perceive(self, **kwargs):
                type(self).calls += 1
                return [SemanticDetection('refrigerator', (2.0, 0.0, 0.5))]

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_FakePerception(),
            semantic_detection_mode='isaac',
        )

        runtime.update(0, _camera_observation())

        labels = sorted(node.label for node in runtime.map.scene_graph.object_nodes())
        self.assertEqual(labels, ['obstacle'])
        self.assertEqual(_FakePerception.calls, 0)
        self.assertEqual(runtime.open_vocabulary_frames, 0)
        self.assertEqual(runtime.statistics()['semantic_detection_mode'], 'isaac')

    def test_open_vocabulary_mode_drops_ground_truth_labels(self):
        class _FakePerception:
            def perceive(self, **kwargs):
                return [SemanticDetection('refrigerator', (2.0, 0.0, 0.5))]

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_FakePerception(),
            semantic_detection_mode='open-vocab',
        )

        runtime.update(0, _camera_observation())

        # The camera's own 'obstacle' box is simulator ground truth, so an
        # open-vocabulary run must not leak it into the scene graph.
        labels = [node.label for node in runtime.map.scene_graph.object_nodes()]
        self.assertEqual(labels, ['refrigerator'])
        self.assertEqual(runtime.open_vocabulary_frames, 1)
        self.assertEqual(runtime.statistics()['semantic_detection_mode'], 'open_vocab')

    def test_open_vocabulary_mode_failure_does_not_fall_back_to_ground_truth(self):
        class _FailingPerception:
            def perceive(self, **kwargs):
                raise RuntimeError('gdino service is down')

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_FailingPerception(),
            semantic_detection_mode='open_vocab',
        )

        runtime.update(0, _camera_observation())

        self.assertEqual(list(runtime.map.scene_graph.object_nodes()), [])
        self.assertEqual(runtime.open_vocabulary_failures, 1)
        self.assertEqual(
            runtime.statistics()['open_vocabulary_last_error'],
            {
                'step': 0,
                'type': 'RuntimeError',
                'message': 'gdino service is down',
            },
        )

    def test_open_vocabulary_mode_stops_after_repeated_runtime_failures(self):
        class _FailingPerception:
            def perceive(self, **kwargs):
                raise RuntimeError('mobile-sam connection refused')

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_FailingPerception(),
            semantic_detection_mode='open_vocab',
            open_vocabulary_failure_limit=3,
        )

        runtime.update(0, _camera_observation())
        runtime.update(24, _camera_observation())
        with self.assertRaisesRegex(RuntimeError, 'failed 3 consecutive frames'):
            runtime.update(48, _camera_observation())

        stats = runtime.statistics()
        self.assertEqual(stats['open_vocabulary_attempts'], 3)
        self.assertEqual(stats['open_vocabulary_failures'], 3)
        self.assertEqual(stats['open_vocabulary_consecutive_failures'], 3)

    def test_rate_limited_open_vocabulary_frame_is_reported_separately(self):
        class _RateLimitedPerception:
            last_query_status = 'rate_limited'
            last_item_errors = []

            def perceive(self, **kwargs):
                return []

        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=True,
            use_rgb_occupancy=False,
            use_semantic_occupancy=False,
            semantic_target='refrigerator',
            open_vocabulary_perception=_RateLimitedPerception(),
            semantic_detection_mode='open_vocab',
        )

        runtime.update(0, _camera_observation())

        stats = runtime.statistics()
        self.assertEqual(stats['open_vocabulary_attempts'], 1)
        self.assertEqual(stats['open_vocabulary_frames'], 0)
        self.assertEqual(stats['open_vocabulary_rate_limited_frames'], 1)

    def test_rgb_depth_endpoints_add_obstacles_without_clearing_rays(self):
        runtime = MapNavigationRuntime(
            mapping_config=MappingConfig(
                x_limits=(0.0, 4.0),
                y_limits=(-2.0, 2.0),
                safe_recovery_y_limits=(-1.5, 1.5),
            ),
            use_scene_graph=False,
            use_rgb_occupancy=True,
            rgb_clears_free_space=False,
        )
        camera = {
            'rgba': np.full((1, 2, 4), 200, dtype=np.uint8),
            'depth': np.ones((1, 2), dtype=np.float32),
            'pointcloud': np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32),
            'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
        }

        runtime.update(0, {'position': np.array([0.0, 0.0, 0.8]), 'sensors': {'camera': camera}})

        for x in (1.0, 2.0):
            cell = runtime.map.occupancy.world_to_cell((x, 0.0))
            self.assertTrue(runtime.map.occupancy.occupied_mask()[cell])

    def test_static_furniture_box_is_inflated_for_planning(self):
        occupancy = OccupancyGridMap(
            MappingConfig(
                x_limits=(0.0, 5.0),
                y_limits=(-2.0, 2.0),
                obstacle_inflation_radius=0.4,
                safe_recovery_y_limits=(-1.5, 1.5),
            )
        )

        occupancy.mark_box_occupied((2.0, -0.5), (3.0, 0.5))
        occupancy.mark_free((2.5, 0.0), radius=0.5)

        self.assertTrue(occupancy.occupied_mask()[occupancy.world_to_cell((2.5, 0.0))])
        self.assertTrue(occupancy.inflated_mask()[occupancy.world_to_cell((1.7, 0.0))])

    def test_static_and_dynamic_obstacles_use_independent_inflation(self):
        occupancy = OccupancyGridMap(
            MappingConfig(
                x_limits=(0.0, 6.0),
                y_limits=(-2.0, 2.0),
                obstacle_inflation_radius=0.2,
                static_obstacle_inflation_radius=0.4,
                safe_recovery_y_limits=(-1.5, 1.5),
            )
        )
        occupancy.mark_box_occupied((2.0, -0.1), (2.1, 0.1))
        occupancy.mark_occupied((4.0, 0.0), radius=0.05, evidence=4.0)
        inflated = occupancy.inflated_mask()

        self.assertTrue(inflated[occupancy.world_to_cell((1.7, 0.0))])
        self.assertFalse(inflated[occupancy.world_to_cell((3.6, 0.0))])

    def test_navigation_runtime_replaces_configured_path_with_map_plan(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        runtime = MapNavigationRuntime(mapping_config=config)
        runtime.map.lidar_frames = 1
        runtime.map.occupancy.observed.fill(True)
        runtime.map.occupancy.log_odds[7:10, 7] = 2.0
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_OBJECT,
            command=ControllerCommand('move_along_path', (((3.0, 0.0, 1.05),),)),
        )

        action = runtime.action_for(
            decision,
            {
                'position': (0.5, 0.0, 1.05),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
        )

        planned_path = action['move_along_path'][0]
        self.assertEqual(planned_path[-1], (3.0, 0.0, 1.05))
        self.assertAlmostEqual(planned_path[-2][0], 2.55)
        self.assertAlmostEqual(planned_path[-2][1], 0.0)
        self.assertEqual(runtime.map.plan_count, 1)
        self.assertNotEqual(planned_path, decision.command.data[0])
        for waypoint in planned_path[:-1]:
            action = runtime.action_for(
                decision,
                {
                    'position': waypoint,
                    'orientation': (1.0, 0.0, 0.0, 0.0),
                },
            )
        self.assertEqual(action['move_along_path'][0], (planned_path[-1],))

    def test_periodic_replanning_uses_latest_occupancy(self):
        config = MappingConfig(
            x_limits=(0.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.25,
            robot_radius=0.1,
        )
        runtime = MapNavigationRuntime(mapping_config=config, replan_interval_steps=120)
        runtime.map.lidar_frames = 1
        runtime.map.occupancy.observed.fill(True)
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_OBJECT,
            command=ControllerCommand('move_along_path', (((3.0, 0.0, 1.05),),)),
        )
        observation = {
            'position': (0.5, 0.0, 1.05),
            'orientation': (1.0, 0.0, 0.0, 0.0),
        }

        runtime.action_for(decision, observation, step=0)
        runtime.action_for(decision, observation, step=120)

        self.assertEqual(runtime.map.plan_count, 2)
        self.assertEqual(runtime.replan_count, 1)

    def test_push_controller_aligns_heading_before_advancing(self):
        runtime = MapNavigationRuntime()
        decision = StateMachineDecision(
            state=InteractionState.PUSH_OBSTACLE,
            command=ControllerCommand('move_by_speed', (0.45, 0.0, 0.0)),
        )

        turning = runtime.action_for(
            decision,
            {
                'orientation': (
                    np.cos(np.pi / 4),
                    0.0,
                    0.0,
                    np.sin(np.pi / 4),
                )
            },
        )
        aligned = runtime.action_for(decision, {'orientation': (1.0, 0.0, 0.0, 0.0)})

        self.assertEqual(turning['move_by_speed'][0], 0.0)
        self.assertLess(turning['move_by_speed'][2], 0.0)
        self.assertEqual(aligned['move_by_speed'], [0.45, 0.0, 0.0])

    def test_object_standoff_aligns_heading_before_ik_transition(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_OBJECT,
            command=ControllerCommand('move_along_path', (((2.4, -0.62, 1.05),),)),
        )

        action = runtime.velocity_action_for(
            decision,
            {
                'position': (2.4, -0.62, 1.05),
                'orientation': (
                    np.cos(np.pi / 4),
                    0.0,
                    0.0,
                    np.sin(np.pi / 4),
                ),
            },
            max_forward_speed=0.35,
            max_lateral_speed=0.18,
        )

        self.assertEqual(action['move_by_speed'][0], 0.0)
        self.assertLess(action['move_by_speed'][2], 0.0)
        self.assertEqual(runtime.map.plan_count, 0)

    def test_scene_graph_pedestal_selects_wide_door_clearance_route(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        runtime.map.occupancy.observed.fill(True)
        detections = [
            SemanticDetection('pedestal', (2.6, -0.64, 0.6)),
            SemanticDetection('pedestal', (2.6, -0.64, 0.6)),
        ]
        runtime.map.scene_graph.update_detections(detections)
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_DOOR,
            command=ControllerCommand('move_along_path', (((4.65, -0.1, 1.05),),)),
        )

        action = runtime.action_for(
            decision,
            {
                'position': (2.4, -0.62, 1.05),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
        )

        path = action['move_along_path'][0]
        self.assertEqual(path[-1], (4.65, -0.1, 1.05))
        self.assertTrue(any(point[1] > 0.0 for point in path[:-1]))

    def test_open_door_scene_graph_creates_centered_portal_waypoint(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        runtime.map.occupancy.observed.fill(True)
        detections = [
            SemanticDetection('door_frame_hinge', (4.9, 0.7, 1.0)),
            SemanticDetection('door_frame_hinge', (4.9, 0.7, 1.0)),
            SemanticDetection('door_frame_latch', (4.9, -0.7, 1.0)),
            SemanticDetection('door_frame_latch', (4.9, -0.7, 1.0)),
        ]
        runtime.map.scene_graph.update_detections(detections)
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_GOAL,
            command=ControllerCommand('move_along_path', (((7.0, 0.0, 1.05),),)),
        )

        action = runtime.action_for(
            decision,
            {
                'position': (4.5, -0.2, 1.05),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
        )

        path = action['move_along_path'][0]
        self.assertAlmostEqual(path[0][0], 5.6)
        self.assertAlmostEqual(path[0][1], 0.0)
        self.assertEqual(path[-1], (7.0, 0.0, 1.05))
        for left, right in zip(path, path[1:]):
            self.assertLessEqual(np.linalg.norm(np.asarray(right[:2]) - np.asarray(left[:2])), 0.45 + 1e-6)

    def test_recovery_uses_latest_safe_mapped_position(self):
        runtime = MapNavigationRuntime()
        runtime.map.occupancy.observed.fill(True)
        runtime.update(
            0,
            {
                'position': (5.6, 0.1, 1.0),
                'sensors': {},
            },
        )
        runtime._planned_state = InteractionState.NAVIGATE_TO_OBJECT
        decision = StateMachineDecision(
            state=InteractionState.RECOVER,
            command=ControllerCommand(
                'recover',
                ((4.65, -0.1, 1.05), (1.0, 0.0, 0.0, 0.0)),
            ),
        )

        action = runtime.action_for(decision, {'position': (5.8, 0.2, 0.4)})

        self.assertEqual(action['recover'][0], (5.6, 0.1, 1.05))

    def test_runtime_static_obstacles_survive_free_space_updates(self):
        config = MappingConfig(
            x_limits=(0.0, 5.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.1,
            robot_radius=0.1,
            static_obstacle_inflation_radius=0.5,
        )
        runtime = MapNavigationRuntime(mapping_config=config)
        runtime.seed_static_obstacles(
            (
                {
                    'label': 'table_0',
                    'minimum_xy': (2.0, -0.5),
                    'maximum_xy': (3.0, 0.5),
                },
            )
        )
        center = runtime.map.occupancy.world_to_cell((2.5, 0.0))
        inflated = runtime.map.occupancy.world_to_cell((1.6, 0.0))

        runtime.update(0, {'position': (2.5, 0.0, 1.0), 'sensors': {}})

        statistics = runtime.statistics()
        self.assertTrue(runtime.map.occupancy.static_occupied[center])
        self.assertTrue(runtime.map.occupancy.inflated_mask()[inflated])
        self.assertEqual(statistics['static_obstacle_boxes'], 1)
        self.assertEqual(statistics['trajectory_static_obstacle_violations'], 1)
        self.assertEqual(statistics['violated_static_obstacle_labels'], ['table_0'])

    def test_carry_navigation_tracks_calibrated_safe_path(self):
        config = MappingConfig(
            x_limits=(-1.0, 4.0),
            y_limits=(-2.0, 2.0),
            grid_resolution=0.1,
            robot_radius=0.1,
        )
        runtime = MapNavigationRuntime(mapping_config=config)
        runtime.map.lidar_frames = 1
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_CARRY_GOAL,
            command=ControllerCommand(
                'move_along_path',
                (((1.0, 0.0, 1.0), (3.0, 0.0, 1.0)),),
            ),
        )
        start_observation = {
            'position': (0.0, 0.0, 1.0),
            'orientation': (1.0, 0.0, 0.0, 0.0),
        }
        near_egress_observation = {
            'position': (0.9, 0.0, 1.0),
            'orientation': (1.0, 0.0, 0.0, 0.0),
        }
        egress_observation = {
            'position': (1.0, 0.0, 1.0),
            'orientation': (1.0, 0.0, 0.0, 0.0),
        }

        initial_action = runtime.action_for(decision, start_observation, step=0)
        near_egress_action = runtime.action_for(decision, near_egress_observation, step=1)
        final_leg_action = runtime.action_for(decision, egress_observation, step=2)

        self.assertEqual(
            initial_action,
            {
                'move_along_path': [
                    ((1.0, 0.0, 1.0), (3.0, 0.0, 1.0))
                ]
            },
        )
        self.assertEqual(near_egress_action, initial_action)
        self.assertEqual(
            final_leg_action,
            {'move_along_path': [((3.0, 0.0, 1.0),)]},
        )
        self.assertEqual(runtime.statistics()['planned_states'], ['navigate_to_carry_goal'])

    def test_calibrated_navigation_velocity_respects_profile_limits(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_OBJECT,
            command=ControllerCommand(
                'move_along_path',
                (((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),),
            ),
        )
        observation = {
            'position': (0.0, 0.0, 1.0),
            'orientation': (
                np.cos(np.pi / 4),
                0.0,
                0.0,
                np.sin(np.pi / 4),
            ),
        }

        action = runtime.velocity_action_for(
            decision,
            observation,
            step=0,
            max_forward_speed=0.35,
            max_lateral_speed=0.18,
        )
        turning_action = runtime.velocity_action_for(
            decision,
            {
                'position': (0.0, 0.0, 1.0),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
            step=1,
            max_forward_speed=0.35,
            max_lateral_speed=0.18,
        )

        self.assertNotIn('move_along_path', action)
        self.assertGreater(action['move_by_speed'][0], 0.0)
        self.assertLessEqual(action['move_by_speed'][0], 0.35)
        self.assertLessEqual(abs(action['move_by_speed'][1]), 0.18)
        self.assertLessEqual(abs(action['move_by_speed'][2]), 1.2)
        self.assertEqual(turning_action['move_by_speed'][:2], [0.0, 0.0])
        self.assertGreater(turning_action['move_by_speed'][2], 0.0)

    def test_object_final_pose_servo_tracks_position_and_fixed_heading(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_OBJECT,
            command=ControllerCommand(
                'move_along_path',
                (((0.2, 0.1, 1.0),),),
            ),
        )

        aligned_action = runtime.velocity_action_for(
            decision,
            {
                'position': (0.0, 0.0, 1.0),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
            step=0,
            max_forward_speed=0.35,
            max_lateral_speed=0.18,
        )
        turning_action = runtime.velocity_action_for(
            decision,
            {
                'position': (0.0, 0.0, 1.0),
                'orientation': (
                    np.cos(np.pi / 4),
                    0.0,
                    0.0,
                    np.sin(np.pi / 4),
                ),
            },
            step=1,
            max_forward_speed=0.35,
            max_lateral_speed=0.18,
        )

        self.assertGreater(aligned_action['move_by_speed'][0], 0.0)
        self.assertGreater(aligned_action['move_by_speed'][1], 0.0)
        self.assertEqual(aligned_action['move_by_speed'][2], 0.0)
        self.assertEqual(turning_action['move_by_speed'][:2], [0.0, 0.0])
        self.assertLess(turning_action['move_by_speed'][2], 0.0)

    def test_carry_reverses_when_the_next_waypoint_is_behind(self):
        runtime = MapNavigationRuntime()
        runtime.map.lidar_frames = 1
        decision = StateMachineDecision(
            state=InteractionState.NAVIGATE_TO_CARRY_GOAL,
            command=ControllerCommand(
                'move_along_path',
                (((0.0, 0.0, 1.0), (-1.0, 0.0, 1.0)),),
            ),
        )

        action = runtime.velocity_action_for(
            decision,
            {
                'position': (0.0, 0.0, 1.0),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
            step=0,
        )

        self.assertLess(action['move_by_speed'][0], 0.0)
        self.assertEqual(action['move_by_speed'][1], 0.0)
        self.assertNotEqual(action['move_by_speed'][2], 0.0)

    def test_object_final_pose_servo_overcomes_locomotion_deadzone(self):
        action = MapNavigationRuntime._pose_target_action(
            (0.05, 0.0, 1.0),
            {
                'position': (0.0, 0.0, 1.0),
                'orientation': (1.0, 0.0, 0.0, 0.0),
            },
            desired_yaw=0.0,
        )

        translation = np.asarray(action['move_by_speed'][:2])
        self.assertAlmostEqual(float(np.linalg.norm(translation)), 0.08)
        self.assertEqual(action['move_by_speed'][2], 0.0)

    def test_door_push_aligns_base_without_dropping_ik_action(self):
        runtime = MapNavigationRuntime()
        decision = StateMachineDecision(
            state=InteractionState.PUSH_DOOR,
            command=ControllerCommand('right_arm_ik_controller', ((5.0, 0.0, 1.0), None)),
            support_commands=(ControllerCommand('move_by_speed', (0.2, 0.0, 0.0)),),
        )

        action = runtime.action_for(
            decision,
            {
                'orientation': (
                    np.cos(np.pi / 4),
                    0.0,
                    0.0,
                    np.sin(np.pi / 4),
                )
            },
        )

        self.assertEqual(action['move_by_speed'][0], 0.0)
        self.assertLess(action['move_by_speed'][2], 0.0)
        self.assertIn('right_arm_ik_controller', action)

    def test_goal_recovery_uses_open_door_portal(self):
        runtime = MapNavigationRuntime()
        runtime._last_safe_position = (4.6, -0.7, 1.0)
        runtime._planned_state = InteractionState.NAVIGATE_TO_GOAL
        runtime._planned_path = ((5.6, 0.0, 1.05), (7.0, 0.0, 1.05))
        runtime.map.traversable_semantic_labels.add('door')
        decision = StateMachineDecision(
            state=InteractionState.RECOVER,
            command=ControllerCommand(
                'recover',
                ((4.65, -0.1, 1.05), (1.0, 0.0, 0.0, 0.0)),
            ),
        )

        action = runtime.action_for(decision, {'position': (4.7, -0.7, 0.4)})

        self.assertEqual(action['recover'][0], (5.6, 0.0, 1.05))

    def test_goal_recovery_keeps_safe_position_after_portal(self):
        runtime = MapNavigationRuntime()
        runtime._last_safe_position = (6.0, 0.1, 1.0)
        runtime._planned_state = InteractionState.NAVIGATE_TO_GOAL
        runtime._planned_path = ((5.6, 0.0, 1.05), (7.0, 0.0, 1.05))
        runtime.map.traversable_semantic_labels.add('door')
        decision = StateMachineDecision(
            state=InteractionState.RECOVER,
            command=ControllerCommand(
                'recover',
                ((4.65, -0.1, 1.05), (1.0, 0.0, 0.0, 0.0)),
            ),
        )

        action = runtime.action_for(decision, {'position': (6.1, 0.2, 0.4)})

        self.assertEqual(action['recover'][0], (6.0, 0.1, 1.05))


def _camera_observation():
    """A one-pixel camera frame whose Isaac semantics label one obstacle."""

    return {
        'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
        'sensors': {
            'camera': {
                'rgba': np.full((1, 1, 4), 200, dtype=np.uint8),
                'depth': np.ones((1, 1), dtype=np.float32),
                'pointcloud': np.array([[2.0, 0.0, 0.5]], dtype=np.float32),
                'position': np.array([0.0, 0.0, 0.8], dtype=np.float32),
                'bounding_box_2d_tight': {
                    'data': np.array([[1, 0, 0, 1, 1, 0.0]], dtype=np.float32),
                    'info': {'idToLabels': {'1': {'class': 'obstacle'}}},
                },
            }
        },
    }


if __name__ == '__main__':
    unittest.main()
