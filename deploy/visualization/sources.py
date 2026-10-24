"""Bounded latest-frame inputs. No navigation, SportClient, or motor publishers."""
import base64
import multiprocessing as mp
import queue
import threading
import time

import cv2
import numpy as np

from deploy.robot.sensors import decode_cloud, decode_image
from deploy.robot.localization import quaternion


def jpeg_from_bgr(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected a decoded colour image')
    h, w = image.shape[:2]
    scale = min(1., 960 / w, 540 / h)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('JPEG encode failed')
    return base64.b64encode(encoded).decode('ascii')


class LatestFrames:
    """One immutable payload per stream; old data remains visible but is marked stale."""
    def __init__(self, max_points=12000, stale_after=.4, demo=False):
        self.lock = threading.Lock()
        self.max_points, self.stale_after, self.demo = max_points, stale_after, demo
        self.streams = {name: None for name in ('cloud', 'pose', 'rgb')}
        self.errors = {}

    def error(self, name, error):
        with self.lock:
            self.errors[name] = str(error)

    def put(self, name, payload, received=None):
        with self.lock:
            self.streams[name] = (time.monotonic() if received is None else received, payload)
            self.errors.pop(name, None)

    def cloud(self, points, frame, stamp=None, source_age=None):
        if not frame:
            raise ValueError('Point cloud frame_id is empty')
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected N x 3 points')
        points = points[np.isfinite(points).all(axis=1)]
        count = len(points)
        if not count:
            raise ValueError('No finite point cloud returns')
        if count > self.max_points:
            points = points[np.linspace(0, count - 1, self.max_points, dtype=int)]
        self.put('cloud', {'points': np.round(points.astype(float), 3).tolist(),
                           'count': count, 'frame': frame, 'stamp': stamp, 'source_age': source_age})

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            result = {'demo': self.demo, 'errors': dict(self.errors)}
            for name, item in self.streams.items():
                if item is None:
                    result[name] = None
                    continue
                received, payload = item
                age = max(0., now - received)
                source_age = payload.get('source_age')
                stale = age > self.stale_after or name in self.errors
                if source_age is not None:
                    stale = stale or source_age + age > self.stale_after or source_age < -.05
                result[name] = dict(payload, age=round(age, 3), stale=bool(stale))
        cloud, pose = result['cloud'], result['pose']
        # No TF guessing and no overlay of mismatched-time/mismatched-frame poses.
        result['pose_overlay'] = bool(cloud and pose and not cloud['stale'] and not pose['stale']
                                      and cloud['frame'] == pose['frame']
                                      and cloud.get('stamp') is not None and pose.get('stamp') is not None
                                      and abs(cloud['stamp'] - pose['stamp']) <= .1)
        return result


class RosViewer:
    """Independent subscriptions, so the viewer also works before calibration/run.py."""
    def __init__(self, frames, config, fps, rgb_topic=None):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2, Image
        from nav_msgs.msg import Odometry
        self.rclpy, self.frames = rclpy, frames
        self.interval, self.last = 1 / fps, {}
        rclpy.init()
        self.node = rclpy.create_node('inter_nav_viewer')
        try:
            self.node.create_subscription(PointCloud2, config['lidar_topic'], self.guard('cloud', self.cloud), qos_profile_sensor_data)
            self.node.create_subscription(Odometry, config['odom_topic'], self.guard('pose', self.pose), qos_profile_sensor_data)
            if rgb_topic:
                self.node.create_subscription(Image, rgb_topic, self.guard('rgb', self.rgb), qos_profile_sensor_data)
        except Exception:
            self.node.destroy_node()
            rclpy.shutdown()
            raise
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()

    def spin(self):
        try:
            self.rclpy.spin(self.node)
        except Exception as error:
            if self.rclpy.ok():
                for name in ('cloud', 'pose', 'rgb'):
                    self.frames.error(name, f'ROS subscription stopped: {error}')

    def guard(self, name, callback):
        def receive(message):
            now = time.monotonic()
            if now - self.last.get(name, float('-inf')) < self.interval and name != 'pose':
                return
            self.last[name] = now
            try:
                callback(message)
            except Exception as error:
                self.frames.error(name, error)
        return receive

    def timing(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            raise ValueError('Missing ROS acquisition timestamp')
        return stamp, self.node.get_clock().now().nanoseconds * 1e-9 - stamp

    def cloud(self, message):
        stamp, age = self.timing(message)
        # Points already have the advertised frame. Do not apply odometry twice.
        self.frames.cloud(decode_cloud(message), message.header.frame_id, stamp, age)

    def pose(self, message):
        stamp, age = self.timing(message)
        p, q = message.pose.pose.position, message.pose.pose.orientation
        w, x, y, z = quaternion([q.w, q.x, q.y, q.z])
        if not np.isfinite([p.x, p.y, p.z]).all() or not message.header.frame_id:
            raise ValueError('Invalid odometry pose/frame')
        self.frames.put('pose', {'position': [p.x, p.y, p.z],
                                'yaw': float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                                'frame': message.header.frame_id, 'stamp': stamp, 'source_age': age})

    def rgb(self, message):
        stamp, age = self.timing(message)
        if message.encoding not in ('rgb8', 'rgba8', 'bgr8', 'bgra8'):
            raise ValueError('Viewer RGB input must be a colour sensor_msgs/Image')
        rgb = decode_image(message)[..., :3]
        self.frames.put('rgb', {'jpeg': jpeg_from_bgr(rgb[..., ::-1]), 'stamp': stamp,
                               'source_age': age, 'source': 'ROS Image'})

    def close(self):
        if self.rclpy.ok():
            self.rclpy.shutdown()
        self.thread.join(timeout=2)
        self.node.destroy_node()


def _offer_latest(output, item):
    try:
        output.put_nowait(item)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        try:
            output.put_nowait(item)
        except queue.Full:
            pass


def _video_worker(output, stop, interface, fps):
    # Spawn isolates the SDK's DDS runtime from ROS Foxy's DDS libraries.
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
        ChannelFactoryInitialize(0, interface)
        client = VideoClient()
        client.SetTimeout(1.)
        client.Init()
        while not stop.is_set():
            started = time.monotonic()
            try:
                code, data = client.GetImageSample()
                if code != 0:
                    raise RuntimeError(f'VideoClient.GetImageSample returned {code}')
                image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
                jpeg = jpeg_from_bgr(image)
                # SDK supplies no acquisition timestamp: label receipt time honestly.
                _offer_latest(output, (time.monotonic(), {'jpeg': jpeg, 'source': 'Go2 VideoClient (receipt time)',
                                                         'stamp': None, 'source_age': None}, None))
            except Exception as error:
                _offer_latest(output, (time.monotonic(), None, str(error)))
            stop.wait(max(0., 1 / fps - (time.monotonic() - started)))
    except Exception as error:
        _offer_latest(output, (time.monotonic(), None, f'{type(error).__name__}: {error}'))


class SdkVideo:
    def __init__(self, frames, interface, fps):
        self.frames = frames
        ctx = mp.get_context('spawn')
        self.output, self.stop = ctx.Queue(maxsize=1), ctx.Event()
        self.process = ctx.Process(target=_video_worker, args=(self.output, self.stop, interface, fps), daemon=True)
        self.process.start()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        while not self.stop.is_set():
            try:
                received, payload, error = self.output.get(timeout=.2)
            except queue.Empty:
                if not self.process.is_alive():
                    if 'rgb' not in self.frames.snapshot()['errors']:
                        self.frames.error('rgb', 'Camera worker stopped; check SDK installation and video service')
                    return
                continue
            if error:
                self.frames.error('rgb', error)
            else:
                self.frames.put('rgb', payload, received)

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.thread.join(timeout=1)
        self.output.close()


class DemoSource:
    """Deterministic synthetic input, always labelled in the UI."""
    def __init__(self, frames, fps):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(frames, fps), daemon=True)
        self.thread.start()

    def run(self, frames, fps):
        rng = np.random.default_rng(7)
        floor = np.column_stack((rng.uniform(-5, 5, (6000, 2)), np.zeros(6000)))
        wall = np.column_stack((np.full(3000, 4.), rng.uniform(-5, 5, 3000), rng.uniform(0, 2, 3000)))
        box = np.column_stack((rng.uniform(.5, 1.5, 2000), rng.uniform(.5, 1.5, 2000), rng.uniform(0, 1, 2000)))
        points = np.concatenate((floor, wall, box))
        tick = 0
        while not self.stop.is_set():
            stamp = time.monotonic()
            frames.cloud(points, 'demo_odom', stamp, 0.)
            frames.put('pose', {'position': [0., 0., .3], 'yaw': .15*np.sin(tick/10),
                                'frame': 'demo_odom', 'stamp': stamp, 'source_age': 0.})
            image = np.full((540, 960, 3), (36, 28, 21), dtype=np.uint8)
            cv2.rectangle(image, (80, 180), (880, 500), (66, 73, 80), -1)
            cv2.rectangle(image, (520, 220), (690, 400), (96, 170, 220), -1)
            cv2.putText(image, 'SYNTHETIC PREVIEW - NO ROBOT', (70, 85), cv2.FONT_HERSHEY_SIMPLEX, .9, (240, 240, 240), 2)
            cv2.putText(image, f'Frame {tick:05d}', (70, 130), cv2.FONT_HERSHEY_SIMPLEX, .7, (150, 200, 160), 2)
            frames.put('rgb', {'jpeg': jpeg_from_bgr(image), 'source': 'Synthetic demo', 'stamp': stamp, 'source_age': 0.})
            tick += 1
            self.stop.wait(1 / fps)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
