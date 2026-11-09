"""Focused contract tests; no model weights or simulator required."""
import unittest
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig, SceneGraphMap, SemanticDetection
from grutopia_extension.interactive_navigation.open_vocabulary_perception import OpenVocabularyPerception, OpenVocabularyPerceptionConfig, AgentVLMBackend, TargetObservationCue
from grutopia_extension.interactive_navigation.semantic_exploration_component import SemanticExplorationComponent, SemanticExplorationConfig
try:
    from .test_open_vocabulary_perception import _Session
except ImportError:  # unittest discover loads tests as top-level modules.
    from test_open_vocabulary_perception import _Session


class _Classifier:
    def __init__(self, label='television'):
        self.label, self.calls = label, []

    def classify(self, image, labels):
        self.calls.append((image.copy(), labels))
        return {'label': self.label, 'model': 'Qwen/Qwen3-VL-8B-Instruct', 'raw_text': '{"label":"' + self.label + '"}'}


class QwenVLPerceptionTest(unittest.TestCase):
    def make_pipeline(self, label='television', clock=lambda: 0, **options):
        self.captions, self.sam_calls = [], []
        classifier = _Classifier(label)
        def detector(image, caption):
            self.captions.append(caption)
            # The same box in both passes should be classified only once.
            return [('monitor', .8, [20, 20, 80, 80])]
        def segment(image, bbox):
            self.sam_calls.append(bbox)
            mask = np.zeros((100, 100), bool)
            mask[20:80, 20:80] = True
            return mask
        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl', **options),
            detector=detector, segmenter=segment, classifier=classifier,
            embedder=lambda image: self.fail('Qwen mode must not load or call CLIP'), clock=clock,
        )
        rgb = np.full((100, 100, 3), 7, np.uint8)
        points = np.zeros((100, 100, 3), np.float32)
        points[20:80, 20:80] = (2, 3, 4)
        return perception, classifier, rgb, np.ones((100, 100)), points

    def test_two_pass_proposals_and_qwen_category_reach_map(self):
        p, classifier, rgb, depth, points = self.make_pipeline()
        result = p.perceive(rgb, depth, points, 'television', step=7)
        self.assertEqual(self.captions[1], 'monitor . screen')
        self.assertNotIn('television', self.captions[0])
        self.assertNotIn('monitor', self.captions[0])
        self.assertIn('mirror', self.captions[0])
        self.assertEqual(len(self.sam_calls), 1)
        self.assertEqual(len(classifier.calls), 1)
        detection = result[0]
        self.assertEqual((detection.raw_label, detection.label, detection.proposal_source), ('monitor', 'television', 'target'))
        self.assertEqual(detection.to_semantic_detection().sources, ('open_vocabulary', 'qwen_vl'))
        self.assertEqual(detection.centroid, (2, 3, 4))
        self.assertEqual(len(p.last_debug_frame['boxes']), 2)
        self.assertEqual(p.last_debug_frame['classified'][0]['classifier_model'], 'Qwen/Qwen3-VL-8B-Instruct')
        crop = classifier.calls[0][0]
        self.assertGreater(crop.shape[0], 60)
        self.assertEqual(int(crop[0, 0, 0]), 7)  # real background retained
        self.assertTrue(np.any(np.all(crop == (255, 0, 0), axis=2)))

    def test_unknown_is_recorded_but_not_mapping_evidence(self):
        p, _, rgb, depth, points = self.make_pipeline('unknown')
        self.assertEqual(p.perceive(rgb, depth, points, 'television'), [])
        self.assertEqual(p.last_debug_frame['classified'][0]['label'], 'unknown')
        self.assertFalse(p.last_debug_frame['classified'][0]['accepted'])
        self.assertEqual(p.last_debug_frame['mapped'], [])
        cue = p.last_target_cues[0]
        self.assertEqual((cue.target_query, cue.classified_label, cue.position), ('television', 'unknown', (2, 3, 4)))
        self.assertEqual(p.last_debug_frame['target_cues'][0]['centroid'], [2, 3, 4])

    def test_rate_limit_does_not_replay_classifications(self):
        p, classifier, rgb, depth, points = self.make_pipeline()
        self.assertEqual(len(p.perceive(rgb, depth, points, 'television', step=1)), 1)
        self.assertEqual(p.perceive(rgb, depth, points, 'television', step=2), [])
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(p.last_debug_frame['classified'], [])
        self.assertEqual(p.last_target_cues, [])

    def cue_component(self, target='television', **options):
        from types import SimpleNamespace
        perception = SimpleNamespace(last_query_status='ok', last_target_cues=[])
        component = SemanticExplorationComponent(SemanticExplorationConfig(
            target_query=target, target_semantic_classifier='qwen-vl',
            mapping=MappingConfig(x_limits=(0, 6), y_limits=(0, 4),
                                  grid_resolution=.1, robot_radius=.1), **options,
        ), perception=perception)
        component.mapping.map.occupancy.observed[:] = True
        component.mapping.update = lambda step, observation: None
        observation = {'position': (1., 1.5, .4), 'orientation': (1., 0., 0., 0.)}
        def frame(step, label='unknown', position=(3., 1.5, .7)):
            perception.last_target_cues = [TargetObservationCue(target, 'screen', label, position, .4, step)]
            component.update(step, observation)
        return component, perception, observation, frame

    def test_cue_guides_without_class_vote_or_success_and_expires(self):
        component, p, observation, frame = self.cue_component()
        frame(0)
        component.update(1, observation)  # Reusing a capture is not a second observation.
        self.assertEqual(component._cue_tracks[0].observations, 1)
        self.assertIsNone(component.target_cue)
        frame(24)
        self.assertEqual(component.state, 'navigate_to_target_cue')
        self.assertIsNone(component.target_node)
        self.assertFalse(component.target_confirmed)
        self.assertFalse(component.mapping.map.scene_graph.object_nodes())
        near = dict(observation, position=component.current_goal)
        component.update(25, near)
        self.assertEqual(component.state, 'observe_target_cue')
        self.assertFalse(component.evaluate(25, near).success)
        self.assertEqual(component.action(25, near)['move_by_speed'][:2], [0., 0.])
        component.update(121, near)
        self.assertIsNone(component.target_cue)
        self.assertEqual(component.cue_history[-1]['reason'], 'unconfirmed_after_observation')
        frame(144)
        self.assertIsNone(component.target_cue)  # Cooldown cannot be bypassed by a new frame.
        self.assertEqual(component._cue_tracks[0].observations, 0)

    def test_qwen_node_preempts_cue_without_relaxing_confirmation(self):
        component, _, observation, frame = self.cue_component(target='plant')
        frame(0)
        frame(24)
        for step in (25, 26):
            component.mapping.map.scene_graph.update_detections([SemanticDetection(
                'plant', (4, 1.5, .7), confidence=.8, step=step, sources=('qwen_vl',))])
        component.update(26, observation)
        self.assertIsNone(component.target_cue)
        self.assertEqual(component.target_node.label, 'plant')
        self.assertFalse(component.target_confirmed)
        for step in range(27, 33):
            component.mapping.map.scene_graph.update_detections([SemanticDetection(
                'plant', (4, 1.5, .7), confidence=.8, step=step, sources=('qwen_vl',))])
        component.update(32, observation)
        self.assertTrue(component.target_confirmed)
        self.assertTrue(component.evaluate(32, dict(observation, position=component.current_goal)).success)

    def test_stale_navigation_and_disabled_cues(self):
        component, _, observation, frame = self.cue_component(target_cue_stale_steps=48)
        frame(0)
        frame(24)
        component.update(73, observation)
        self.assertIsNone(component.target_cue)
        self.assertEqual(component.cue_history[-1]['reason'], 'stale')
        component, _, observation, frame = self.cue_component(target_cue_navigation_steps=48)
        frame(0)
        frame(24)
        frame(72)
        self.assertIsNone(component.target_cue)
        self.assertEqual(component.cue_history[-1]['reason'], 'navigation_timeout')
        component, _, _, frame = self.cue_component(enable_target_cues=False)
        frame(0)
        frame(24)
        self.assertEqual(component._cue_tracks, [])

    def test_candidate_budget_prioritizes_target_pass(self):
        p, *_ = self.make_pipeline(qwen_vl_max_candidates=1)
        selected = p._select_proposals([('chair', .99, (0, 0, 10, 10), 'context'), ('screen', .4, (30, 30, 60, 60), 'target')], 0)
        self.assertEqual(selected[0][3], 'target')

    def test_health_checks_vl_without_loading_clip(self):
        session = _Session(get_payloads=[{'ready': True}] * 3)
        backend = AgentVLMBackend(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'), session=session)
        p = OpenVocabularyPerception(backend.config, detector=backend, segmenter=backend, classifier=backend)
        p._ensure_clip = lambda: self.fail('CLIP loading in Qwen mode')
        self.assertIn('qwen_vl', p.check_ready()['services'])
        self.assertEqual(len(session.get_calls), 3)

    def test_qwen_classes_do_not_merge_by_embedding_or_count_same_frame_twice(self):
        graph = SceneGraphMap(MappingConfig())
        def detection(label, step, confidence=.8):
            return SemanticDetection(label=label, position=(1, 1, 1), confidence=confidence, step=step,
                                     embedding=(1., 0.), sources=('open_vocabulary', 'qwen_vl'))
        graph.update_detections([detection('monitor', 1), detection('television', 1), detection('television', 1, .9)])
        nodes = {n.label: n for n in graph.object_nodes()}
        self.assertEqual(set(nodes), {'monitor', 'television'})
        self.assertEqual(nodes['television'].observations, 1)
        graph.update_detections([detection('television', 1)])
        self.assertEqual(nodes['television'].observations, 1)
        graph.update_detections([detection('television', 2)])
        self.assertEqual(graph.label_counts(nodes['television'].node_id)['television'], 2)

    def test_target_keeps_existing_vote_requirement_without_extra_review(self):
        component = SemanticExplorationComponent(SemanticExplorationConfig(target_query='television', target_semantic_classifier='qwen-vl'))
        for step in range(8):
            component.mapping.map.scene_graph.update_detections([SemanticDetection(
                label='tv', position=(1,1,1), confidence=.8, step=step, sources=('qwen_vl',))])
        component.target_node = component._best_target_node()
        self.assertIsNotNone(component.target_node)
        self.assertTrue(component.target_confirmed)
        other = SemanticExplorationComponent(SemanticExplorationConfig(target_query='television', target_semantic_classifier='qwen-vl'))
        for step in range(8):
            other.mapping.map.scene_graph.update_detections([SemanticDetection(
                label='monitor', position=(1,1,1), confidence=.8, step=step, sources=('qwen_vl',))])
        self.assertIsNone(other._best_target_node())
