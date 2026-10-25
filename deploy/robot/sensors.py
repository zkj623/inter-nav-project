"""ROS 2 input adapter. Existing drivers/SLAM own acquisition and TF."""
import threading
import time
import numpy as np
from .localization import PoseContinuity, PoseHistory, camera_cloud, multiply, transform, quaternion
from .observation import Snapshot, normalize


def decode_image(message):
    formats = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3), 'rgba8': ('u1', 4),
               'bgra8': ('u1', 4), '16UC1': ('u2', 1), '32FC1': ('f4', 1)}
    if message.encoding not in formats:
        raise ValueError(f'Unsupported image encoding {message.encoding}')
    dtype, channels = formats[message.encoding]
    dtype = np.dtype(('>' if message.is_bigendian else '<') + dtype)
    row_bytes = message.width * channels * dtype.itemsize
    if message.step < row_bytes or len(message.data) < message.height * message.step:
        raise ValueError('Invalid image stride or payload')
    image = np.ndarray((message.height, message.width, channels), dtype=dtype,
                       buffer=bytes(message.data), strides=(message.step, channels*dtype.itemsize, dtype.itemsize)).copy()
    if channels == 1:
        image = image[..., 0].astype(np.float32)
        if message.encoding == '16UC1':
            image *= .001
    elif message.encoding in ('bgr8', 'bgra8'):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return image


def decode_cloud(message, *, strip_utlidar_padding=False):
    """Foxy-compatible PointCloud2 XYZ reader; honour fields, stride and byte order."""
    if message.width <= 0 or message.height <= 0 or message.point_step <= 0:
        raise ValueError('Empty or invalid PointCloud2 dimensions')
    if (message.row_step < message.width*message.point_step
            or len(message.data) != message.height*message.row_step):
        raise ValueError('Invalid PointCloud2 stride or payload size')
    fields = {field.name: field for field in message.fields}
    columns = []
    payload = bytes(message.data)
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in (7, 8):
            raise ValueError('PointCloud2 XYZ must be scalar FLOAT32 or FLOAT64 fields')
        dtype = np.dtype(('>' if message.is_bigendian else '<') + ('f4' if field.datatype == 7 else 'f8'))
        if field.offset < 0 or field.offset+dtype.itemsize > message.point_step:
            raise ValueError('PointCloud2 field exceeds point_step')
        columns.append(np.ndarray((message.height, message.width), dtype=dtype,
                                  buffer=payload, offset=field.offset,
                                  strides=(message.row_step, message.point_step)).reshape(-1))
    points = np.column_stack(columns).astype(np.float32)
    # Observed on /utlidar/cloud_deskewed: 10,000 zero XYZ slots precede
    # the actual scan. Match the raw prefix BEFORE finite filtering/TF.
    # Do not discard isolated world-origin returns or zeros on other topics.
    if strip_utlidar_padding and len(points) >= 10000 and np.all(points[:10000] == 0):
        points = points[10000:]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('PointCloud2 contains no finite returns')
    return points


class RosSensors:
    def __init__(self, config, preview_rgb=None, preview_topic=None):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import PointCloud2, Image, CameraInfo
        from nav_msgs.msg import Odometry
        from tf2_ros import Buffer, TransformListener
        self.config, self.rclpy, self.Time = config, rclpy, Time
        rclpy.init()
        self.node = rclpy.create_node('inter_nav_deploy')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.lock = threading.Lock()
        self.pose = self.lidar = self.camera = None
        self.images = {}
        self.last_error = None
        self.sequence = 0
        self.pose_history = PoseHistory(config['pose_cloud_tolerance_s'])
        self.last_stamps = {}
        self._errors = {}
        self.continuity = PoseContinuity(config['max_position_jump_m'], config['max_orientation_jump_rad'])
        for topic, cls, callback in (
            ('odom_topic', Odometry, self._pose), ('lidar_topic', PointCloud2, self._lidar),
            ('rgb_topic', Image, lambda msg: self._image('rgb', msg)),
            ('depth_topic', Image, lambda msg: self._image('depth', msg)),
            ('camera_info_topic', CameraInfo, lambda msg: self._image('info', msg)),
        ):
            if topic in ('rgb_topic', 'depth_topic', 'camera_info_topic') and config['camera_mode'] != 'rgbd':
                continue
            self.node.create_subscription(cls, config[topic], self._guard(callback, topic), qos_profile_sensor_data)
        if preview_rgb is not None and preview_topic:
            last_preview = [float('-inf')]
            def preview(message):
                now = time.monotonic()
                if now-last_preview[0] < .5:
                    return
                last_preview[0] = now
                _, captured_at = self._stamp(message)
                rgb = decode_image(message)
                if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                    raise ValueError('Preview requires a colour sensor_msgs/Image')
                preview_rgb(rgb, captured_at)
            self.node.create_subscription(Image, preview_topic, self._guard(preview), qos_profile_sensor_data)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    @property
    def last_error(self):
        errors = getattr(self, '_errors', {}).copy()
        return '; '.join(errors.values()) or getattr(self, '_pair_error', None)

    @last_error.setter
    def last_error(self, value):
        self._pair_error = value

    def _guard(self, callback, key='preview'):
        def invoke(message):
            if not hasattr(self, '_errors'):
                self._errors = {}
            try:
                callback(message)
            except Exception as error:
                self._errors[key] = f'{key}: {type(error).__name__}: {error}'
            else:
                self._errors.pop(key, None)
        return invoke

    def _stamp(self, message):
        stamp = message.header.stamp
        seconds = stamp.sec + stamp.nanosec*1e-9
        age = self.node.get_clock().now().nanoseconds*1e-9 - seconds
        if seconds <= 0:
            raise ValueError('ROS timestamp invalid')
        if age < -.05:
            raise ValueError(f'ROS clock mismatch: message ahead of Orin by {-age*1000:.1f} ms (limit 50 ms); use sync_time.py --watch before run')
        if age > self.config['max_data_age_s']:
            raise ValueError(f'ROS timestamp stale: age={age*1000:.1f} ms, limit={self.config["max_data_age_s"]*1000:.0f} ms; check clocks and receive delay')
        return stamp.sec*1_000_000_000 + stamp.nanosec, time.monotonic()-max(0, age)

    def _tf(self, frame, stamp):
        if not frame:
            raise ValueError('Empty sensor coordinate frame')
        if frame == self.config['map_frame']:
            return np.zeros(3), np.array([1., 0, 0, 0])
        tf = self.buffer.lookup_transform(self.config['map_frame'], frame, self.Time.from_msg(stamp)).transform
        return np.array([tf.translation.x, tf.translation.y, tf.translation.z]), np.array(
            [tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z])

    def _pose(self, message):
        seq, received = self._stamp(message)
        if message.child_frame_id != self.config['base_frame']:
            raise ValueError('Odometry child_frame_id must equal base_frame')
        if seq <= self.last_stamps.get('pose', -1):
            raise ValueError('Non-increasing odometry timestamp')
        # Odometry already contains parent->child pose; TF publication is optional.
        pose = message.pose.pose
        p = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = quaternion([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        if not np.isfinite(p).all():
            raise ValueError('Invalid odometry position')
        parent_p, parent_q = self._tf(message.header.frame_id, message.header.stamp)
        p, q = transform(p, parent_p, parent_q), multiply(parent_q, q)
        v = message.twist.twist
        velocity = (v.linear.x, v.linear.y, v.angular.z)
        if not np.isfinite(velocity).all():
            raise ValueError('Invalid odometry velocity')
        self.continuity.check(p, q)
        with self.lock:
            self.pose_history.add(seq, p, q)
            self.last_stamps['pose'] = seq
            self.pose = (seq, received, p, q, velocity)

    def _lidar(self, message):
        seq, received = self._stamp(message)
        p, q = self._tf(message.header.frame_id, message.header.stamp)
        if seq <= self.last_stamps.get('lidar', -1):
            raise ValueError('Non-increasing point-cloud timestamp')
        points = decode_cloud(message, strip_utlidar_padding=(
            self.config['lidar_topic'] == '/utlidar/cloud_deskewed'
            and message.header.frame_id == 'odom'))
        points = transform(points, p, q).astype(np.float32)
        with self.lock:
            self.last_stamps['lidar'] = seq
            self.lidar = (received, {'pointcloud': points, 'physics_step': seq,
                                     'max_range': self.config['lidar_max_range']})

    def _image(self, kind, message):
        if kind != 'info':
            self._stamp(message)
        self.images[kind] = message
        if set(self.images) != {'rgb', 'depth', 'info'}:
            return
        rgb, depth, info = (self.images[k] for k in ('rgb', 'depth', 'info'))
        rgb_seq, rgb_at = self._stamp(rgb)
        depth_seq, depth_at = self._stamp(depth)
        if abs(rgb_seq-depth_seq)*1e-9 > self.config['image_sync_tolerance_s']:
            return
        if rgb.header.frame_id != depth.header.frame_id or info.header.frame_id != depth.header.frame_id:
            raise ValueError('Require rectified RGB and aligned depth in the same optical frame')
        if (rgb.width, rgb.height) != (info.width, info.height):
            raise ValueError('CameraInfo resolution mismatch')
        p, q = self._tf(depth.header.frame_id, depth.header.stamp)
        rgba, metres = decode_image(rgb), decode_image(depth)
        if rgba.ndim != 3 or metres.ndim != 2 or rgba.shape[:2] != metres.shape:
            raise ValueError('RGB and depth must be pixel-aligned')
        projection = np.asarray(info.p, dtype=float).reshape(3, 4)[:, :3]
        k = projection if projection[0, 0] > 0 and projection[1, 1] > 0 else np.asarray(info.k).reshape(3, 3)
        xyz = camera_cloud(metres, k, p, q)
        camera = {'rgba': rgba, 'depth': metres, 'pointcloud': xyz, 'position': p,
                  'orientation': multiply(q, (.5, .5, -.5, .5)),
                  'horizontal_fov_degrees': float(np.rad2deg(2*np.arctan(info.width/(2*k[0, 0])))),
                  'capture_step': depth_seq}
        with self.lock:
            self.camera = (min(rgb_at, depth_at), camera)

    def latest(self):
        if self.continuity.fault:
            raise ValueError(self.continuity.fault)
        with self.lock:
            if self.pose is None or self.lidar is None:
                return None
            seq, pose_at, p, q, velocity = self.pose
            lidar_at, lidar = self.lidar
            lidar = dict(lidar)
            try:
                cloud_p, cloud_q = self.pose_history.at(lidar['physics_step'])
            except ValueError as error:
                self.last_error = str(error)
                return None
            self._pair_error = None
            offset = self.config['lidar_origin_offset_base_m']
            calibrated = offset is not None and self.config['ground_z'] is not None
            # Missing calibration permits recording only; runtime must not build a map.
            lidar['position'] = transform(offset if offset is not None else [0, 0, 0], cloud_p, cloud_q)
            sensors = {'lidar': lidar}
            camera_at = None
            if self.camera is not None:
                camera_at, sensors['camera'] = self.camera
            observation = normalize({'position': p, 'orientation': q, 'sensors': sensors}, self.config['ground_z'] if calibrated else 0.)
        self.sequence += 1
        return Snapshot(observation, time.monotonic(), pose_at, lidar_at, camera_at, velocity, self.sequence,
                        geometry_calibrated=calibrated, planning_frame=self.config['map_frame'])

    def close(self):
        self.rclpy.shutdown()
        self.thread.join(timeout=2)
        self.node.destroy_node()
