"""Read-only scene worker: own display voxels and render detached map snapshots."""
import signal
import time

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, VoxelPointCloudMap
from .scene import display_points, map_payload
from .sources import jpeg_from_bgr


class FrameRenderer:
    """One ordered input stream; navigation never reads this object's voxels."""
    def __init__(self, config, separate_cloud):
        self.separate_cloud = separate_cloud
        self.voxels = VoxelPointCloudMap(MappingConfig(**config)) if separate_cloud else None
        self.cloud_step = None
        self.cloud_at = None

    def render(self, item):
        started = time.monotonic()
        scan = item['scan']
        if scan is not None and self.separate_cloud:
            step = scan['physics_step']
            if self.cloud_step is not None and step < self.cloud_step:
                raise ValueError('Display cloud received an out-of-order scan')
            if step != self.cloud_step:
                # Match the release mapper's measured-return contract. This
                # historical display layer does not decide collision freedom;
                # current hit/free evidence lives in OnlineNavigationMap.
                points = np.asarray(scan['pointcloud'], dtype=float)
                distance = np.linalg.norm(points-np.asarray(scan['position']), axis=1)
                measured = points[np.isfinite(points).all(axis=1) & (distance < scan['max_range']*.99)]
                self.voxels.update(measured, step=scan['step'], source='lidar')
                self.cloud_step, self.cloud_at = step, scan['captured_at']
        cloud_finished = time.monotonic()
        if item['scene'] is None:
            return None
        snapshot = item['snapshot']
        scene = dict(item['scene'])
        if scene['ready'] and self.separate_cloud:
            if self.cloud_step is None:
                raise ValueError('Display cloud has not received an accepted scan')
            scene.update(points=display_points(self.voxels), voxel_count=self.voxels.voxel_count,
                         cloud_updated_at=self.cloud_at, cloud_physics_step=self.cloud_step)
        lidar = snapshot.observation['sensors']['lidar']
        points = scene['points'] if scene['ready'] else lidar['pointcloud']
        if len(points) > 12000:
            points = points[np.linspace(0, len(points)-1, 12000, dtype=int)]
        cloud_at = scene.get('cloud_updated_at', scene['updated_at']) if scene['ready'] else snapshot.lidar_at
        cloud = {'points': np.round(np.asarray(points, dtype=float), 3).tolist(),
                 'count': scene['voxel_count'] if scene['ready'] else len(lidar['pointcloud']),
                 'display_count': len(points), 'frame': snapshot.planning_frame,
                 'accumulated': bool(scene['ready']), 'stamp': cloud_at, 'source_age': None,
                 'role': 'display_history_only',
                 'ground_relative_z': snapshot.geometry_calibrated,
                 'kind': '累计建图点云' if scene['ready'] else '当前原始点云（未建图）'}
        w, x, y, z = snapshot.observation['orientation']
        pose = {'position': snapshot.observation['position'].tolist(),
                'yaw': float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                'frame': snapshot.planning_frame, 'stamp': snapshot.pose_at, 'source_age': None}
        camera = snapshot.observation['sensors'].get('camera')
        rgb = None
        if camera is not None:
            rgb = {'jpeg': jpeg_from_bgr(camera['rgba'][..., :3][..., ::-1]),
                   'source': 'ROS RGB-D colour', 'stamp': snapshot.camera_at, 'source_age': None,
                   'timestamp_basis': 'monotonic_capture_estimate'}
        mapping = map_payload(scene)
        finished_at = time.monotonic()
        return {'snapshot': snapshot, 'scene': scene, 'map': mapping, 'cloud': cloud,
                'pose': pose, 'rgb': rgb, 'finished_at': finished_at,
                'timings': {'capture_ms': item['capture_ms'],
                            'queue_ms': (started-item['queued_at'])*1000,
                            'cloud_update_ms': (cloud_finished-started)*1000,
                            'encode_ms': (finished_at-cloud_finished)*1000}}


def process_frames(inputs, outputs, config, separate_cloud):
    """Spawn target. Parent owns the input queue, result pipe and shutdown deadline."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        renderer = FrameRenderer(config, separate_cloud)
    except Exception as error:
        outputs.send({'error': f'{type(error).__name__}: {error}', 'startup': True})
        return
    outputs.send({'ready': True})
    while True:
        item = inputs.get()
        if item is None:
            outputs.send(None)
            return
        try:
            result = renderer.render(item)
            if result is not None:
                outputs.send(result)
        except Exception as error:
            outputs.send({'error': f'{type(error).__name__}: {error}',
                         'frame_dropped': item.get('scene') is not None,
                         'cloud_failed': item.get('scan') is not None})
