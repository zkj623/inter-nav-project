"""Task state adapter. Existing mapping/planning implementations remain authoritative."""
import math
from contextlib import contextmanager
from time import perf_counter
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig, canonical_semantic_label
from grutopia_extension.interactive_navigation.mapping_runtime import SemanticDetectionMode
from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent, PointNavigationConfig
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent, SemanticExplorationConfig,
)
from grutopia_extension.interactive_navigation.visual_observation import VisualObservationMemory


class Tasks:
    def __init__(self, config, perception=None, scorer=None, *, maintain_voxels=True):
        self.config, self.perception = config, perception
        self.scorer = scorer
        self.mapping_config = MappingConfig(**config['mapping'])
        self.component = self._point((0., 0., .3))
        self.component.mapping.map.maintain_voxels = maintain_voxels
        self.mode, self.state = 'observe', 'observe'
        self.elapsed = 0.
        self.last_time = None
        self.last_step = -1
        self.last_lidar = self.last_camera = None
        self.result = None
        self.label_baseline = {}
        self.map_ready = False
        self._observe_tick = -1
        self._maintenance_due = None
        self.last_mapping_at = None
        self.last_timings = {}

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
        # Reuse the shared distance taper with a deployment-specific radius.
        # Success is evaluated before the next action, then requests StopMove.
        component.mapping.final_goal_slow_radius = c['final_goal_slow_radius']
        component.mapping.defer_topology_updates = True
        component.mapping.topology_update_interval = max(1, round(c['topology_interval_s']*c['tick_hz']))
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
        if mode == 'explore' and not self.component.mapping.map.maintain_voxels:
            raise ValueError('Exploration requires the full map; restart with RGB-D configured')
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
        if self.perception is not None and (mode == 'explore' or self.mode == 'explore'):
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
        if self.perception is not None and self.mode == 'explore':
            self.perception.reset()
        self.state = self.mode = 'observe'
        self.result = None
        self.elapsed, self.last_time = 0., None
        self.component.mapping.invalidate_path()
        self.component.mapping.semantic_target = ''
        if hasattr(self.component, 'target_node'):
            self.component.target_node = None

    def fail(self, reason):
        self.state = 'fault'
        self.result = {'success': False, 'reason': reason}

    def check_navigation_goal(self, goal):
        """Check the requested cell; route planning remains in the original planner."""
        if len(goal) < 2 or not all(math.isfinite(v) for v in goal[:2]):
            raise ValueError('目标坐标必须为有限数值')
        if not self.map_ready:
            raise ValueError('导航地图尚未就绪，请先观察建图')
        grid = self.component.mapping.navigation_occupancy
        cell = grid.world_to_cell(goal[:2])
        prefix = f'目标 ({goal[0]:.2f}, {goal[1]:.2f})'
        if cell is None:
            raise ValueError(f'{prefix} 超出地图范围')
        if not grid.observed[cell]:
            raise ValueError(f'{prefix} 尚未观测，请先观察该区域')
        if grid.occupied_mask()[cell]:
            raise ValueError(f'{prefix} 位于地图障碍内')
        if grid.inflated_mask()[cell]:
            raise ValueError(f'{prefix} 位于障碍膨胀区，当前地图净空不足')

    @contextmanager
    def _measure(self, stage):
        started = perf_counter()
        try:
            yield
        finally:
            self.last_timings[stage] = perf_counter()-started

    def update(self, snapshot, now):
        self.last_timings = {}
        with self._measure('total_s'):
            return self._update(snapshot, now)

    def _update(self, snapshot, now):
        if not snapshot.geometry_calibrated:
            if self.state == 'active':
                raise ValueError('Navigation requires calibrated ground and LiDAR origin')
            return None
        observation = {**snapshot.observation, 'sensors': dict(snapshot.observation['sensors'])}
        lidar = observation['sensors'].get('lidar', {})
        lidar_id = lidar.get('physics_step')
        if lidar_id == self.last_lidar:
            observation['sensors'].pop('lidar', None)
        camera = observation['sensors'].get('camera')
        if camera is not None:
            frame = camera['capture_step']
            if frame == self.last_camera:
                observation['sensors'].pop('camera', None)
            else:
                if self.perception is not None:
                    self.perception.set_frame(frame, snapshot.camera_at)
        if self.state != 'active':
            # Continue sensing while paused/manual; no task elapsed time advances.
            due = self._maintenance_due is None or now >= self._maintenance_due
            if due:
                self._maintenance_due = now + self.config['topology_interval_s']
            self._observe_tick += 1
            with self._measure('mapping_s'):
                self.component.mapping.update(self._observe_tick, observation, maintenance=due)
            self._accept_observation(observation, snapshot)
            return None
        self.elapsed += max(0, now-self.last_time)
        self.last_time = now
        step = max(self.last_step+1, int(self.elapsed*self.config['tick_hz']))
        self.last_step = step
        if self.elapsed >= self.config['task_timeout_s']:
            self.state = 'completed'
            self.result = {'success': False, 'reason': 'task_timeout'}
            return None
        with self._measure('mapping_s'):
            self.component.update(step, observation)
        self._accept_observation(observation, snapshot)
        if self.mode == 'navigate':
            with self._measure('goal_check_s'):
                self.check_navigation_goal(self.component.goal)
        with self._measure('evaluation_s'):
            result = self.component.evaluate(step, observation)
        if result.success and self.mode == 'explore' and not self.fresh_target_confirmed():
            # Retained semantic maps can guide a new task, but cannot prove a fresh find.
            return np.zeros(3)
        if result.terminal:
            self.state = 'completed'
            self.result = result.as_event()
            return None
        mapping = self.component.mapping
        failures = mapping.planning_failures
        with self._measure('planning_tracking_s'):
            action = self.component.action(step, observation)
        if self.mode == 'navigate' and mapping.planning_failures > failures:
            # A zero action alone is not a failure (e.g. recovery or turning).
            # Surface only a new error from the existing planner, not stale history.
            raise ValueError(f'当前地图无法规划到目标：{mapping.last_planning_error}')
        if set(action) != {'move_by_speed'}:
            raise ValueError(f'Unsupported real-robot action: {list(action)}')
        return np.asarray(action['move_by_speed'], dtype=float)

    def _accept_observation(self, observation, snapshot):
        # Only deduplicate inputs that the algorithm actually processed successfully.
        sensors = observation['sensors']
        if 'lidar' in sensors:
            self.last_lidar = sensors['lidar']['physics_step']
            self.last_mapping_at = snapshot.lidar_at
            self.map_ready = True
        if 'camera' in sensors:
            self.last_camera = sensors['camera']['capture_step']
