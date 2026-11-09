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
                      path=_payload(engine.path), skeleton=[],
                      frontiers=[] if frontier is None else [list(frontier)],
                      objects=[{'position': [*box.xy, box.center_z or box.dimensions[2]/2],
                                'label': box.id, 'dimensions': list(box.dimensions), 'yaw': box.yaw}
                               for box in layers.current_boxes][:40],
                      goal=None if engine.goal is None else list(engine.goal),
                      phase=engine.phase, decision=decision,
                      skill=None if skill is None else {'state': skill.state, 'status': skill.status,
                          'reason': skill.reason, 'object_id': skill.plan.box_id},
                      cloud_role='display_history_only')
        # Display voxels belong exclusively to the rendering process. Capturing
        # a map never extracts or changes a navigation-side point cloud.
        return result


def render_map(scene, size=720):
    """Same occupancy/path conventions as simulation; equal scale on X and Y."""
    observed, occupied, inflated = scene['observed'], scene['occupied'], scene['inflated']
    grid = np.full((*observed.shape, 3), (37, 33, 29), dtype=np.uint8)
    grid[observed] = (210, 217, 220)
    grid[inflated & observed & ~occupied] = (115, 185, 228)
    grid[occupied] = (75, 75, 215)
    if 'tracked' in scene:
        grid[scene['tracked']] = (180, 80, 150)
    for row, col in scene['skeleton']:
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = (210, 135, 50)
    height, width = grid.shape[:2]
    scale = min(size / width, size / height)
    dw, dh = round(width * scale), round(height * scale)
    left, top = (size-dw)//2, (size-dh)//2
    frame = np.full((size, size, 3), (27, 20, 13), dtype=np.uint8)
    frame[top:top+dh, left:left+dw] = cv2.resize(np.flipud(grid), (dw, dh), interpolation=cv2.INTER_NEAREST)
    config = scene['config']

    def pixel(p):
        col = (p[0]-config['x_limits'][0])/config['grid_resolution']
        row = (p[1]-config['y_limits'][0])/config['grid_resolution']
        return int(round(left+col*dw/width)), int(round(top+dh-row*dh/height))

    for path, color, thickness in ((scene['trajectory'], (70, 160, 55), 2),
                                    ([scene['position'], *scene['path']], (245, 130, 30), 3)):
        if len(path) >= 2:
            cv2.polylines(frame, [np.array([pixel(p) for p in path], dtype=np.int32)], False, color, thickness)
    for p in scene['frontiers']:
        cv2.circle(frame, pixel(p), 4, (220, 210, 40), -1)
    for obj in scene['objects']:
        p = pixel(obj['position'])
        if 'dimensions' in obj:
            half = np.asarray(obj['dimensions'][:2])/2
            angle = np.radians(obj['yaw'])
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*half
            corners = corners @ rotation.T + obj['position'][:2]
            cv2.polylines(frame, [np.array([pixel(c) for c in corners], dtype=np.int32)], True, (180, 80, 150), 2)
        cv2.circle(frame, p, 4, (180, 80, 150), -1)
        cv2.putText(frame, obj['label'][:16], (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .35, (180, 80, 150), 1)
    if scene['goal'] is not None:
        cv2.drawMarker(frame, pixel(scene['goal']), (0, 215, 255), cv2.MARKER_CROSS, 18, 2)
    p = scene['position']
    w, x, y, z = scene['orientation']
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    cv2.circle(frame, pixel(p), 5, (80, 240, 80), -1)
    cv2.arrowedLine(frame, pixel(p), pixel([p[0]+.5*np.cos(yaw), p[1]+.5*np.sin(yaw)]), (20, 120, 20), 2)
    return frame


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
            'observed_cells': int(scene['observed'].sum()),
            'tracked_objects': len(scene['objects']), 'cloud_role': 'display_history_only'}
