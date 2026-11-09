"""Read-only adapters for the same CPU renderers used by simulation videos."""
from types import SimpleNamespace

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner, BoxState
from grutopia_extension.interactive_navigation.task_video_recording import TaskVideoRecorder
from grutopia_extension.interactive_navigation.visualization import InteractionVideoRecorder


# Simulation cloud conventions. This is a display slice, never a mapping filter.
CLOUD_HEIGHT = (.1, .8)
CLOUD_PALETTE = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
CLOUD_STYLE = {'height_range': list(CLOUD_HEIGHT),
               'palette': ['#%02x%02x%02x' % tuple(c[::-1]) for c in CLOUD_PALETTE],
               'background': '#f9f9f9', 'robot': '#4d8c3b'}


class SnapshotGeometry:
    """Only the grid-coordinate and box-mask methods needed to draw a snapshot."""
    cell = BoxPushPlanner.cell
    box_mask = BoxPushPlanner.box_mask

    def __init__(self, scene):
        cfg = scene['config']
        self.limits = (cfg['x_limits'], cfg['y_limits'])
        self.resolution = cfg['grid_resolution']
        self.navigation_half_width = scene.get('navigation_half_width', .22)
        self.free = np.asarray(scene['observed'], bool)


def map_renderer(scene):
    """No planner execution, video writers, Isaac imports or live map references."""
    position = np.asarray(scene['position'])
    w, x, y, z = scene['orientation']
    yaw = float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    blank = np.zeros_like(scene['observed'], bool)
    now = scene.get('map_time_s', 0.)
    layers = SimpleNamespace(
        planner=SnapshotGeometry(scene), observed=scene['observed'],
        static_occupied=scene.get('static', scene['occupied']),
        stable_obstacles=scene.get('stable', blank),
        local_dynamic=scene.get('local_dynamic', scene['occupied'] & ~scene.get('static', blank)),
        tracked_occupied=scene.get('tracked', blank), global_tracked=scene.get('global_tracked', blank),
        vacated_time=np.where(scene.get('recently_vacated', blank), now, -np.inf),
        now=now, robot_xy=position[:2], robot_yaw=yaw,
        local_radius=scene.get('local_radius', 3.),
        local_period=scene.get('local_period', .1), global_period=scene.get('global_period', 1.),
        local_revision=scene.get('revision', 0), global_revision=scene.get('global_revision', 0),
        decision=scene.get('decision') or {},
        interaction_enabled=scene.get('interaction_enabled', False),
        interactable_ids=set(scene.get('interactable_ids', ())),
        recognized_objects=scene.get('recognized_objects', {}),
        current_boxes=[BoxState(obj['label'], tuple(obj['position'][:2]), tuple(obj['dimensions']), obj.get('yaw', 0.))
                       for obj in scene.get('objects', ()) if 'dimensions' in obj])
    # Some archived packets stored a dataclass decision with nested explanation.
    if 'explanation' in layers.decision:
        layers.decision = {**(layers.decision['explanation'] or {}),
                           **{k: v for k, v in layers.decision.items() if k != 'explanation'}}
    runtime = SimpleNamespace(map=SimpleNamespace(config=SimpleNamespace(**scene['config'])),
                              navigation_layers=layers, _planned_path=scene.get('path', ()))
    renderer = TaskVideoRecorder.__new__(TaskVideoRecorder)
    renderer.task = None
    if scene.get('goal') is not None:
        trajectory = scene.get('trajectory', ())
        start = scene.get('task_start')
        if start is None:
            start = trajectory[0][:2] if len(trajectory) else position[:2]
        renderer.task = {'id': 'D01', 'start': start, 'goal': scene['goal'][:2]}
    renderer._trajectory = list(scene.get('trajectory', ()))
    renderer.map_size, renderer.map_plot_width = (1280, 720), 1000
    return renderer, runtime, SimpleNamespace(robot_position=position, robot_yaw=yaw)


def render_navigation_map(scene, size=720):
    renderer, runtime, observation = map_renderer(scene)
    frame = renderer._render_map(0, scene.get('phase', scene.get('state', 'observe')), observation, runtime)
    # Monotonic sensor times are not experiment elapsed times.
    renderer._header(frame, 'Global guidance + live local map | ' + scene.get('state', 'observe') + ' | ' + scene['frame'])
    if size != 720:
        frame = cv2.resize(frame, (round(size*1280/720), size), interpolation=cv2.INTER_AREA)
    return frame


def render_follow_cloud(points, position, orientation, size=720):
    renderer = InteractionVideoRecorder.__new__(InteractionVideoRecorder)
    renderer.map_size, renderer.map_plot_width = (1280, 720), 1000
    renderer._map_viewport = (0, 36, 1000, 684)
    runtime = SimpleNamespace(map=SimpleNamespace(voxels=SimpleNamespace(points=lambda: np.asarray(points))))
    observation = SimpleNamespace(robot_position=np.asarray(position))
    frame = renderer._render_pointcloud(0, 'display history', observation, runtime, {'orientation': orientation})
    if size != 720:
        frame = cv2.resize(frame, (round(size*1280/720), size), interpolation=cv2.INTER_AREA)
    return frame
