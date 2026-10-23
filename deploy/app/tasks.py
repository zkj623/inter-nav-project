"""Task state adapter. Existing mapping/planning implementations remain authoritative."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig, canonical_semantic_label
from grutopia_extension.interactive_navigation.mapping_runtime import SemanticDetectionMode
from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent, SemanticExplorationConfig,
)
from grutopia_extension.interactive_navigation.visual_observation import VisualObservationMemory


class Tasks:
    def __init__(self, config, perception=None, scorer=None):
        self.config, self.perception = config, perception
        self.scorer = scorer
        self.mapping_config = MappingConfig(**config['mapping'])
        self.component = self._point((0., 0., .3))
        self.mode, self.state = 'observe', 'observe'
        self.elapsed = 0.
        self.last_time = None
        self.last_step = -1
        self.last_lidar = self.last_camera = None
        self.result = None
        self.label_baseline = {}
        self.map_ready = False

    def _point(self, goal):
        c = self.config
        component = PointNavigationComponent(PointNavigationConfig(
            goal=tuple(goal), mapping=self.mapping_config, max_steps=2**60,
            velocity_control=True, prefer_voronoi_paths=True, use_scene_graph=False,
            direct_path_tracking_margin=c.get('direct_path_tracking_margin', .10),
            lidar_interval=1, rgb_interval=1, safe_base_height=c['safe_base_height'],
            fall_height=c['fall_height'], success_distance=c['success_distance'],
            max_forward_speed=c['max_forward_speed'], max_lateral_speed=c['max_lateral_speed']))
        # Geometry-only tasks have no detector; never enable Isaac fallback.
        component.mapping.semantic_detection_mode = SemanticDetectionMode.OPEN_VOCABULARY
        return component

    def _explore(self, target):
        if self.perception is None:
            raise ValueError('Exploration requires configured remote perception')
        c = self.config
        ticks = lambda seconds: max(1, int(math.ceil(seconds*c['tick_hz'])))
        return SemanticExplorationComponent(SemanticExplorationConfig(
            target_query=target, mapping=self.mapping_config, max_steps=2**60,
            semantic_detection_mode='open_vocab', target_semantic_classifier='qwen-vl',
            enable_target_cues=False, safe_base_height=c['safe_base_height'], fall_height=c['fall_height'],
            max_forward_speed=c['max_forward_speed'], max_lateral_speed=c['max_lateral_speed'],
            topology_update_interval=ticks(c['topology_interval_s']),
            frontier_selection_interval=ticks(c['frontier_selection_s']),
            frontier_progress_timeout=ticks(c['frontier_progress_timeout_s']),
            frontier_goal_timeout=ticks(c['frontier_goal_timeout_s']),
            frontier_scan_timeout=ticks(c['frontier_scan_timeout_s']),
            visual_observation_budget=ticks(c['visual_observation_budget_s']),
        ), perception=self.perception, scorer=self.scorer)

    def start(self, mode, value, now):
        if mode not in ('navigate', 'explore'):
            raise ValueError('Unknown task')
        if mode == 'navigate' and (len(value) != 3 or not np.isfinite(value).all()):
            raise ValueError('Goal must contain finite map coordinates')
        new = self._point(value) if mode == 'navigate' else self._explore(value)
        # Retain one map and its topology. New task gets fresh path/recovery state.
        previous = self.component.mapping
        new.mapping.map = previous.map
        new.mapping.semantic_voronoi = previous.semantic_voronoi
        new.mapping.voronoi_planner = previous.voronoi_planner
        new.mapping.lidar_interval = new.mapping.rgb_interval = 1
        new.mapping.topology_update_interval = max(1, round(self.config['topology_interval_s']*self.config['tick_hz']))
        new.mapping.buffer_recovery_timeout = max(1, round(5*self.config['tick_hz']))
        if mode == 'explore':
            new.visual_memory = VisualObservationMemory(previous.map.occupancy)
            self.perception.reset()
        self.component, self.mode, self.state = new, mode, 'active'
        self.elapsed, self.last_time, self.last_step = 0., now, -1
        self.result = None
        if mode == 'explore':
            self.label_baseline = {node.node_id: self._matching_votes(node.node_id)
                                   for node in self.component.mapping.map.scene_graph.object_nodes()}

    def _matching_votes(self, node_id):
        query = canonical_semantic_label(self.component.config.target_query.strip().lower())
        counts = self.component.mapping.map.scene_graph.label_counts(node_id)
        return sum(count for label, count in counts.items() if canonical_semantic_label(label) == query)

    def fresh_target_confirmed(self):
        node = self.component.target_node
        if node is None:
            return False
        new_votes = self._matching_votes(node.node_id)-self.label_baseline.get(node.node_id, 0)
        return new_votes >= self.config['fresh_target_observations']

    def pause(self):
        if self.state == 'active':
            self.state = 'paused'

    def resume(self, now):
        if self.state != 'paused':
            raise ValueError('No paused task')
        self.component.mapping.invalidate_path()
        self.last_time = now
        self.state = 'active'

    def cancel(self):
        self.state = 'observe'
        self.component.mapping.invalidate_path()
        self.component.mapping.semantic_target = ''

    def fail(self, reason):
        self.state = 'fault'
        self.result = {'success': False, 'reason': reason}

    def update(self, snapshot, now):
        if not snapshot.geometry_calibrated:
            if self.state == 'active':
                raise ValueError('Navigation requires calibrated ground and LiDAR origin')
            return None
        self.map_ready = True
        observation = {**snapshot.observation, 'sensors': dict(snapshot.observation['sensors'])}
        lidar = observation['sensors'].get('lidar', {})
        lidar_id = lidar.get('physics_step')
        if lidar_id == self.last_lidar:
            observation['sensors'].pop('lidar', None)
        else:
            self.last_lidar = lidar_id
        camera = observation['sensors'].get('camera')
        if camera is not None:
            frame = camera['capture_step']
            if frame == self.last_camera:
                observation['sensors'].pop('camera', None)
            else:
                self.last_camera = frame
                if self.perception is not None:
                    self.perception.set_frame(frame, snapshot.camera_at)
        if self.state != 'active':
            # Continue sensing while paused/manual; no task elapsed time advances.
            self.component.mapping.update(max(0, self.last_step), observation)
            return None
        self.elapsed += max(0, now-self.last_time)
        self.last_time = now
        step = max(self.last_step+1, int(self.elapsed*self.config['tick_hz']))
        self.last_step = step
        if self.elapsed >= self.config['task_timeout_s']:
            self.state = 'completed'
            self.result = {'success': False, 'reason': 'task_timeout'}
            return None
        self.component.update(step, observation)
        result = self.component.evaluate(step, observation)
        if result.success and self.mode == 'explore' and not self.fresh_target_confirmed():
            # Retained semantic maps can guide a new task, but cannot prove a fresh find.
            return np.zeros(3)
        if result.terminal:
            self.state = 'completed'
            self.result = result.as_event()
            return None
        action = self.component.action(step, observation)
        if set(action) != {'move_by_speed'}:
            raise ValueError(f'Unsupported real-robot action: {list(action)}')
        return np.asarray(action['move_by_speed'], dtype=float)
