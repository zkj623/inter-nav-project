"""Copy the running planner's geometry; render it without another mapper."""
from collections import deque
from dataclasses import asdict
import base64

import cv2
import numpy as np


def display_points(voxel_map, limit=12000):
    """Sample through the existing public API; do not alter the navigation map.

    The output/network payload is bounded. points() still extracts all centroids;
    this does not bound the upstream extraction cost on large maps.
    """
    if limit < 1:
        raise ValueError('display point limit must be positive')
    points = voxel_map.points()
    stride = max(1, (len(points)+limit-1)//limit)
    return points[::stride].copy()


class SceneCapture:
    def __init__(self):
        self.trajectory = deque(maxlen=4000)

    def capture(self, snapshot, tasks):
        position = snapshot.observation['position'].copy()
        if not self.trajectory or np.linalg.norm(position - self.trajectory[-1]) > .03:
            self.trajectory.append(position)
        result = {'ready': bool(tasks.map_ready and snapshot.geometry_calibrated),
                  'frame': snapshot.planning_frame, 'position': position,
                  'orientation': snapshot.observation['orientation'].copy(),
                  'trajectory': np.asarray(self.trajectory), 'state': tasks.state, 'mode': tasks.mode,
                  'result': tasks.result}
        if not result['ready']:
            result['reason'] = '地面/雷达原点未标定，或尚未收到有效观测；不生成导航地图'
            return result
        runtime = tasks.component.mapping
        grid = runtime.map.occupancy
        topology = runtime.semantic_voronoi.cached_snapshot() if runtime.semantic_voronoi else None
        result.update(config=asdict(runtime.map.config), revision=grid.revision,
                      observed=grid.observed.copy(), occupied=grid.occupied_mask().copy(),
                      inflated=grid.inflated_mask().copy(), log_odds=grid.log_odds.copy(),
                      static=grid.static_occupied.copy(),
                      points=display_points(runtime.map.voxels),
                      voxel_count=runtime.map.voxels.voxel_count,
                      path=[list(p) for p in (runtime._planned_path or ())],
                      skeleton=[list(cell) for cell in topology.skeleton_cells] if topology else [],
                      frontiers=[list(f.centroid) for f in topology.frontiers] if topology else [],
                      objects=[{'position': list(n.position), 'label': n.label}
                               for n in runtime.map.scene_graph.object_nodes()][:40])
        goal = getattr(tasks.component.config, 'goal', None)
        target = getattr(tasks.component, 'target_node', None)
        result['goal'] = list(target.position) if target is not None else (list(goal) if goal is not None and tasks.mode == 'navigate' else None)
        return result


def render_map(scene, size=720):
    """Same occupancy/path conventions as simulation; equal scale on X and Y."""
    observed, occupied, inflated = scene['observed'], scene['occupied'], scene['inflated']
    grid = np.full((*observed.shape, 3), (37, 33, 29), dtype=np.uint8)
    grid[observed] = (210, 217, 220)
    grid[inflated & ~occupied] = (115, 185, 228)
    grid[occupied] = (75, 75, 215)
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
            'resolution': scene['config']['grid_resolution'], 'path_points': len(scene['path'])}
