import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, SemanticDetection
from grutopia_extension.interactive_navigation.mapping_runtime import _deduplicate_semantic_detections
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent,
    SemanticExplorationConfig,
    SemanticExplorationStatus,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiConfig
from grutopia_extension.interactive_navigation.semantic_run_summary import (
    build_run_summary,
    summary_path,
    write_run_summary,
)


class _FakeTextEmbeddingPerception:
    def embed_text(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _component(target='carry object', frontier_selection_interval=1, perception=None,
               confirmation_observations=2):
    return SemanticExplorationComponent(
        SemanticExplorationConfig(
            target_query=target,
            mapping=MappingConfig(
                x_limits=(0.0, 5.0),
                y_limits=(0.0, 4.0),
                grid_resolution=0.1,
                robot_radius=0.1,
                safe_recovery_y_limits=(0.1, 3.9),
            ),
            voronoi=SemanticVoronoiConfig(spur_length=0.0),
            target_min_observations=2,
            target_confirmation_min_label_observations=confirmation_observations,
            topology_update_interval=1,
            frontier_selection_interval=frontier_selection_interval,
        ),
        perception=perception,
    )


class SemanticExplorationComponentTest(unittest.TestCase):
    def test_detection_mode_reaches_the_mapping_runtime(self):
        component = _component('refrigerator')
        self.assertEqual(component.mapping.semantic_detection_mode.value, 'hybrid')
        self.assertEqual(
            component.statistics()['semantic_detection_effective_mode'],
            'isaac',
        )

        open_vocab = SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query='refrigerator',
                semantic_detection_mode='open_vocab',
            ),
            perception=_FakeTextEmbeddingPerception(),
        )
        self.assertEqual(open_vocab.mapping.semantic_detection_mode.value, 'open_vocab')

    def test_hybrid_preflight_failure_is_persisted_in_statistics(self):
        component = SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query='refrigerator',
                semantic_detection_mode='hybrid',
                open_vocabulary_startup_error='ConnectionError: service offline',
            ),
        )

        stats = component.statistics()
        self.assertEqual(stats['semantic_detection_mode'], 'hybrid')
        self.assertEqual(stats['semantic_detection_effective_mode'], 'isaac')
        self.assertFalse(stats['open_vocabulary_available'])
        self.assertEqual(
            stats['open_vocabulary_startup_error'],
            'ConnectionError: service offline',
        )

    def test_open_vocab_mode_without_perception_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticExplorationComponent(
                SemanticExplorationConfig(
                    target_query='refrigerator',
                    semantic_detection_mode='open_vocab',
                )
            )

    def test_unknown_detection_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticExplorationConfig(target_query='refrigerator', semantic_detection_mode='radar')

    def test_empty_map_waits_for_fresh_initial_scan(self):
        component = _component('refrigerator', frontier_selection_interval=160)
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        for step in range(20):
            component.update(step, observation)

        self.assertEqual(len(component.decision_history), 0)
        self.assertIsNone(component.current_goal)

    def test_unknown_target_selects_reachable_frontier(self):
        component = _component('refrigerator')
        component.mapping.map.occupancy.observed[8:32, 5:30] = True
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)
        self.assertIsNone(component.current_goal)
        observation['sensors']['lidar'] = {'horizontal_fov': 360, 'rotation_frequency': 0}
        component.mapping.map.lidar_frames += 8
        component.update(1, observation)

        self.assertIsNone(component.target_node)
        self.assertIsNotNone(component.current_frontier_id)
        self.assertIsNotNone(component.current_goal)
        self.assertEqual(component.state, 'explore_navigate')

    def test_persistent_semantic_target_switches_to_target_navigation(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7), confidence=0.9),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7), confidence=0.9),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)

        self.assertIsNotNone(component.target_node)
        self.assertEqual(component.state, 'navigate_to_semantic_target')
        self.assertIsNotNone(component.current_goal)
        self.assertNotEqual(component.current_goal[:2], component.target_node.position[:2])

    def test_embedding_target_upgrades_to_lexical_match(self):
        component = _component('refrigerator', perception=_FakeTextEmbeddingPerception())
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        look_alike = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.9, embedding=tuple(look_alike)),
                SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.9, embedding=tuple(look_alike)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)

        self.assertIsNotNone(component.target_node)
        self.assertEqual(component.target_node.label, 'door')
        self.assertEqual(component.statistics()['target_match'], 'embedding')

        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection(
                    'refrigerator',
                    (4.0, 2.5, 0.8),
                    confidence=0.9,
                    sources=('isaac',),
                ),
                SemanticDetection(
                    'refrigerator',
                    (4.0, 2.5, 0.8),
                    confidence=0.9,
                    sources=('isaac',),
                ),
            ]
        )
        component.update(1, observation)

        self.assertEqual(component.target_node.label, 'refrigerator')
        self.assertEqual(component.statistics()['target_match'], 'lexical')
        self.assertEqual(component.statistics()['target_match_method'], 'lexical')
        self.assertEqual(component.statistics()['target_sources'], ['isaac'])
        self.assertIsNotNone(component.current_goal)

    def test_single_relabel_does_not_confirm_refrigerator(self):
        component = _component('refrigerator', perception=_FakeTextEmbeddingPerception())
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        for _ in range(4):
            component.mapping.map.scene_graph.update_detections(
                [SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.2, embedding=(1.0, 0.0, 0.0))]
            )
        component.mapping.map.scene_graph.update_detections(
            [SemanticDetection('refrigerator', (3.0, 1.5, 0.7), confidence=0.9, embedding=(1.0, 0.0, 0.0))]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        self.assertEqual(component.target_node.label, 'refrigerator')
        self.assertEqual(component.statistics()['target_label_support'], 1)
        self.assertFalse(component.target_confirmed)
        near = dict(observation, position=component.current_goal)
        self.assertEqual(component.evaluate(1, near).status, SemanticExplorationStatus.RUNNING)
        component.update(1, near)
        self.assertIsNone(component.target_node)

    def test_repeated_refrigerator_detections_survive_door_majority(self):
        component = _component(
            'refrigerator',
            perception=_FakeTextEmbeddingPerception(),
            confirmation_observations=8,
        )
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        graph = component.mapping.map.scene_graph
        for step in range(30):
            graph.update_detections([SemanticDetection(
                'door', (3.0, 1.5, 0.7), confidence=0.55,
                embedding=(1.0, 0.0), step=step,
            )])
        for step in range(7):
            graph.update_detections([SemanticDetection(
                'refrigerator door', (3.0, 1.5, 0.7), confidence=0.40,
                embedding=(1.0, 0.0), step=30 + step,
            )])
        observation = {
            'position': (1.0, 1.5, 0.4),
            'orientation': (1.0, 0.0, 0.0, 0.0),
            'sensors': {},
        }
        component.update(40, observation)
        self.assertIsNone(component.target_node)
        self.assertFalse(component.target_confirmed)

        graph.update_detections([SemanticDetection(
            'refrigerator door', (3.0, 1.5, 0.7), confidence=0.40,
            embedding=(1.0, 0.0), step=41,
        )])
        component.update(41, observation)
        self.assertEqual(component.target_node.label, 'door')
        self.assertEqual(component.statistics()['target_match'], 'lexical')
        self.assertEqual(component.statistics()['target_label_support'], 8)
        self.assertTrue(component.target_confirmed)
        self.assertEqual(
            component.evaluate(42, dict(observation, position=component.current_goal)).status,
            SemanticExplorationStatus.SUCCEEDED,
        )

    def test_fused_door_frames_confirm_refrigerator_at_eighth_frame(self):
        component = _component('refrigerator', confirmation_observations=8)
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        observation = {
            'position': (1.0, 1.5, 0.4),
            'orientation': (1.0, 0.0, 0.0, 0.0),
            'sensors': {},
        }
        for step in range(8):
            fused = _deduplicate_semantic_detections((
                SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.55,
                                  embedding=(1.0, 0.0), step=step),
                SemanticDetection('refrigerator door', (3.0, 1.5, 0.7),
                                  confidence=0.40, embedding=(1.0, 0.0), step=step),
            ))
            component.mapping.map.scene_graph.update_detections(fused)
            component.update(step, observation)
            self.assertEqual(component.target_confirmed, step == 7)
        self.assertEqual(component.target_node.label, 'door')
        self.assertEqual(component.statistics()['target_label_support'], 8)

    def test_target_switch_prefers_target_labels_over_generic_observations(self):
        component = _component('refrigerator', confirmation_observations=8)
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        graph = component.mapping.map.scene_graph
        graph.update_detections([SemanticDetection(
            'refrigerator door', (3.0, 1.5, 0.7),
            confidence=0.7, embedding=(1.0, 0.0), step=0,
        )])
        for step in range(1, 40):
            graph.update_detections([SemanticDetection(
                'door', (3.0, 1.5, 0.7), confidence=0.5,
                embedding=(1.0, 0.0), step=step,
                label_evidence=('door', 'refrigerator door') if step < 5 else (),
            )])
        observation = {
            'position': (1.0, 1.5, 0.4),
            'orientation': (1.0, 0.0, 0.0, 0.0),
            'sensors': {},
        }
        component.update(40, observation)
        original = component.target_node.node_id
        self.assertEqual(component.statistics()['target_label_support'], 5)
        for step in range(8):
            graph.update_detections([SemanticDetection(
                'refrigerator door', (4.5, 3.0, 0.7), step=41 + step,
            )])
        component.update(49, observation)
        self.assertNotEqual(component.target_node.node_id, original)
        self.assertEqual(component.statistics()['target_label_support'], 8)
        self.assertTrue(component.target_confirmed)

    def test_embedding_only_chair_cannot_complete_refrigerator_query(self):
        component = _component('refrigerator', perception=_FakeTextEmbeddingPerception())
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        for _ in range(2):
            component.mapping.map.scene_graph.update_detections(
                [SemanticDetection('chair', (3.0, 1.5, 0.7), embedding=(1.0, 0.0, 0.0))]
            )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        self.assertEqual(component.target_node.label, 'chair')
        self.assertFalse(component.statistics()['target_confirmed'])
        near_chair = dict(observation, position=component.current_goal)
        self.assertEqual(component.evaluate(1, near_chair).status, SemanticExplorationStatus.RUNNING)

        component.update(1, near_chair)
        self.assertIsNone(component.target_node)
        self.assertEqual(component.statistics()['rejected_provisional_targets'], 1)
        component.update(2, near_chair)
        self.assertIsNone(component.target_node)

        for _ in range(2):
            component.mapping.map.scene_graph.update_detections(
                [SemanticDetection('refrigerator door', (4.0, 2.5, 0.8))]
            )
        component.update(3, near_chair)
        self.assertEqual(component.target_node.label, 'refrigerator door')
        self.assertTrue(component.statistics()['target_confirmed'])
        at_goal = dict(observation, position=component.current_goal)
        self.assertEqual(component.evaluate(4, at_goal).status, SemanticExplorationStatus.SUCCEEDED)

    def test_lexical_target_upgrades_only_to_strictly_better_node(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        locked = component.target_node.node_id

        # A same-label node that merely ties on (confidence, observations)
        # must not steal the lock.
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
            ]
        )
        component.update(1, observation)
        self.assertEqual(component.target_node.node_id, locked)

        # Once the other instance is strictly better observed (the real
        # object outpaces a long-range ghost estimate), the lock upgrades
        # and the approach position is recomputed.
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
            ]
        )
        component.update(2, observation)

        self.assertNotEqual(component.target_node.node_id, locked)
        self.assertEqual(
            tuple(round(value, 2) for value in component.target_node.position[:2]),
            (4.5, 3.0),
        )
        self.assertEqual(component.statistics()['target_match'], 'lexical')

    def test_success_and_persistence_include_semantic_topology(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        reached = dict(observation)
        reached['position'] = component.current_goal

        result = component.evaluate(1, reached)

        self.assertEqual(result.status, SemanticExplorationStatus.SUCCEEDED)
        fallen = dict(reached, position=(*component.current_goal[:2], 0.0))
        self.assertEqual(component.evaluate(1, fallen).failure_reason, 'robot_fell')
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'semantic_map')
            component.save(prefix)
            payload = json.loads(Path(prefix + '.json').read_text(encoding='utf-8'))
            self.assertIn('semantic_voronoi', payload)
            self.assertIn('semantic_exploration', payload)
            self.assertIn('label_counts', payload['scene_graph'])


    def test_compact_events_and_explicit_arrival_summary(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        progress = component.progress_event(0, observation)
        self.assertLessEqual(len(progress), 7)
        self.assertNotIn('statistics', progress)
        self.assertGreater(progress['goal_distance_m'], component.config.target_distance)

        incomplete = build_run_summary(
            target_query=component.config.target_query,
            detection_mode='isaac',
            max_steps=100,
            profile_goal=(1.0, 1.5, 0.4),
            component=component,
            step=0,
            robot_observation=observation,
            stop_reason='interrupted',
            exit_code=130,
        )
        self.assertEqual(incomplete['status'], 'incomplete')
        self.assertFalse(incomplete['arrival_verified'])
        self.assertEqual(incomplete['profile_goal_distance_min_m'], 0.0)
        self.assertEqual(incomplete['profile_goal_distance_final_m'], 0.0)
        self.assertEqual(incomplete['profile_goal_distance_min_step'], 0)
        startup_failure = build_run_summary(
            target_query='refrigerator',
            detection_mode='open_vocab',
            max_steps=100,
            stop_reason='exception',
            error_type='RuntimeError',
            exit_code=2,
        )
        self.assertEqual(startup_failure['status'], 'failed')
        self.assertFalse(startup_failure['target_found'])

        reached = dict(observation, position=component.current_goal)
        unconfirmed = build_run_summary(
            target_query=component.config.target_query,
            detection_mode='isaac',
            max_steps=100,
            component=component,
            step=1,
            robot_observation=reached,
            stop_reason='interrupted',
            exit_code=130,
        )
        self.assertEqual(unconfirmed['status'], 'incomplete')
        self.assertFalse(unconfirmed['arrival_verified'])
        result = component.evaluate(1, reached)
        self.assertNotIn('statistics', result.as_event())
        self.assertEqual(result.as_event()['status'], 'succeeded')
        completed = build_run_summary(
            target_query=component.config.target_query,
            detection_mode='isaac',
            max_steps=100,
            component=component,
            terminal_result=result,
            step=1,
            robot_observation=reached,
            exit_code=0,
        )
        self.assertEqual(completed['status'], 'succeeded')
        self.assertTrue(completed['arrival_verified'])
        self.assertLessEqual(completed['goal_distance_m'], completed['arrival_threshold_m'])
        self.assertEqual(completed['arrival_rule'], 'xy_distance_to_approach_goal')
        self.assertGreaterEqual(completed['arrival_margin_m'], 0.0)
        self.assertIsNotNone(completed['target_distance_m'])
        with tempfile.TemporaryDirectory() as directory:
            path = summary_path(directory, '')
            write_run_summary(path, completed)
            self.assertEqual(json.loads(path.read_text())['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
