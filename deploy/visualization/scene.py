"""Copy the running planner's geometry; render it without another mapper."""
from collections import deque
from dataclasses import asdict, is_dataclass
import base64

import cv2
import numpy as np


def _payload(value):
    """Detach decision diagnostics and keep HTTP JSON finite."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _payload(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_payload(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def display_points(voxel_map, limit=12000):
    """Sample through the existing public API; do not alter the navigation map.

    The output/network payload is bounded. points() still extracts all centroids;
    this does not bound the upstream extraction cost on large maps.
    """
    if limit < 1:
        raise ValueError('display point limit must be positive')
    points = voxel_map.points()
    if len(points) <= limit:
        return points.copy()
    # Stable spatial sampling: changing the total voxel count must not change
    # a global stride and make whole bands of historical geometry blink out.
    keys = np.floor(points / voxel_map.config.voxel_size).astype(np.int64).astype(np.uint64)
    hashes = (keys[:, 0]*np.uint64(73856093) ^ keys[:, 1]*np.uint64(19349663)
              ^ keys[:, 2]*np.uint64(83492791))
    hashes = (hashes ^ (hashes >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    hashes = (hashes ^ (hashes >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    hashes ^= hashes >> np.uint64(31)
    selected = np.argpartition(hashes, limit-1)[:limit]
    return points[np.sort(selected)].copy()


class SceneCapture:
    def __init__(self):
        self.trajectory = deque(maxlen=4000)

    def capture(self, snapshot, tasks):
        position = snapshot.observation['position'].copy()
        if not self.trajectory or np.linalg.norm(position - self.trajectory[-1]) > .03:
            self.trajectory.append(position)
        result = {'ready': bool(tasks.map_ready and snapshot.geometry_calibrated),
                  'updated_at': tasks.last_mapping_at,
                  'frame': snapshot.planning_frame, 'position': position,
                  'orientation': snapshot.observation['orientation'].copy(),
                  'trajectory': np.asarray(self.trajectory), 'state': tasks.state, 'mode': tasks.mode,
                  'result': tasks.result, 'decision_stages_s': dict(tasks.last_timings)}
        perception = getattr(tasks, 'perception', None)
        if perception is not None:
            result['interaction_perception'] = _payload(getattr(perception, 'last_delivery', None))
            result['interaction_perception_error'] = getattr(perception, 'last_error', None)
            delivery = result['interaction_perception']
            result['interaction_observation_age_s'] = (max(0., snapshot.captured_at-delivery['captured_at'])
                if delivery is not None and delivery.get('captured_at') is not None else None)
        if not result['ready']:
            result['reason'] = '地面/雷达原点未标定，或尚未收到有效观测；不生成导航地图'
            return result
        engine = tasks.engine
        layers = engine.layers
        decision = _payload(engine.decision)
        detail = decision.get('explanation', decision) if isinstance(decision, dict) else {}
        frontier = (detail or {}).get('frontier_goal')
        skill = engine.skill
        result.update(config=asdict(layers.config), revision=layers.local_revision,
                      global_revision=layers.global_revision, topology_revision=None,
                      observed=layers.observed.copy(), occupied=layers.local_occupied.copy(),
                      inflated=engine.planner.blocked(layers.current_boxes).copy(),
                      log_odds=layers.evidence.log_odds.copy(),
                      static=layers.static_occupied.copy(), tracked=layers.tracked_occupied.copy(),
                      global_observed=layers.global_observed.copy(),
                      global_occupied=layers.global_occupied.copy(),
                      stable=layers.stable_obstacles.copy(),
                      local_dynamic=layers.local_dynamic.copy(),
                      global_tracked=layers.global_tracked.copy(),
                      recently_vacated=(layers.now-layers.vacated_time < 3.).copy(),
                      map_time_s=layers.now, local_radius=layers.local_radius,
                      local_period=layers.local_period, global_period=layers.global_period,
                      navigation_half_width=getattr(engine.planner, 'navigation_half_width', .22),
                      interaction_enabled=bool(getattr(engine, 'config', {}).get('interaction_enabled', False)),
                      interactable_ids=list(getattr(layers, 'interactable_ids', ())),
                      recognized_objects=_payload(getattr(layers, 'recognized_objects', {})),
                      task_start=_payload(getattr(engine, 'task_start', None)),
                      path=_payload(engine.path), skeleton=[],
                      frontiers=[] if frontier is None else [list(frontier)],
                      objects=[{'position': [*box.xy, box.center_z or box.dimensions[2]/2],
                                'label': box.id, 'dimensions': list(box.dimensions), 'yaw': box.yaw}
                               for box in layers.current_boxes][:40],
                      goal=None if engine.goal is None else list(engine.goal),
                      phase=engine.phase, decision=decision,
                      interaction_source=getattr(engine, 'interaction_source', None),
                      interaction_command=_payload(getattr(engine, 'interaction_command', None)),
                      contact_collision=_payload(engine.contact_collision.diagnostic) if getattr(engine, 'contact_collision', None) is not None else None,
                      skill=None if skill is None else {'state': skill.state, 'status': skill.status,
                          'reason': skill.reason, 'object_id': skill.plan.box_id},
                      cloud_role='display_history_only')
        # Display voxels belong exclusively to the rendering process. Capturing
        # a map never extracts or changes a navigation-side point cloud.
        return result


def render_map(scene, size=720):
    """Render detached navigation layers with the simulation's map renderer."""
    from .simulation_view import render_navigation_map
    return render_navigation_map(scene, size)


def map_payload(scene):
    if not scene['ready']:
        return {'ready': False, 'reason': scene['reason']}
    ok, image = cv2.imencode('.png', render_map(scene))
    if not ok:
        raise ValueError('Map PNG encoding failed')
    return {'ready': True, 'png': base64.b64encode(image).decode(), 'revision': scene['revision'],
            'frame': scene['frame'], 'voxel_count': scene['voxel_count'],
            'resolution': scene['config']['grid_resolution'], 'path_points': len(scene['path']),
            'global_revision': scene.get('global_revision'), 'phase': scene.get('phase'),
            'decision': scene.get('decision'), 'skill': scene.get('skill'),
            'interaction_source': scene.get('interaction_source'),
            'interaction_perception': scene.get('interaction_perception'),
            'interaction_perception_error': scene.get('interaction_perception_error'),
            'interaction_observation_age_s': scene.get('interaction_observation_age_s'),
            'contact_collision': scene.get('contact_collision'),
            'observed_cells': int(scene['observed'].sum()),
            'tracked_objects': len(scene['objects']), 'cloud_role': 'display_history_only'}
