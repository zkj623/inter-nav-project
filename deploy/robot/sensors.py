"""ROS 2 input adapter. Existing drivers/SLAM own acquisition and TF."""
from collections import deque
import threading
import time
import numpy as np
from .localization import (PoseContinuity, PoseHistory, camera_cloud, multiply, transform,
                           rotation, registered_floor_height)
from .observation import Snapshot, normalize
from .timebase import SourceTimeBridge
from .mixed_cloud import LioMount, MixedCloud, odometry_pose


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
    points = np.column_stack(columns).astype(np.float32, copy=False)
    # Observed on /utlidar/cloud_deskewed: 10,000 zero XYZ slots precede
    # the actual scan. Match the raw prefix BEFORE finite filtering/TF.
    # Do not discard isolated world-origin returns or zeros on other topics.
    if strip_utlidar_padding and len(points) >= 10000 and np.all(points[:10000] == 0):
        points = points[10000:]
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
    if not len(points):
        raise ValueError('PointCloud2 contains no finite returns')
    return points


class RosSensors:
    def __init__(self, config, preview_rgb=None, preview_topic=None, *, timebase=None, node_name='inter_nav_deploy'):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import PointCloud2, Image, CameraInfo
        from nav_msgs.msg import Odometry
        from tf2_ros import Buffer, TransformListener
        self.config, self.rclpy, self.Time = dict(config), rclpy, Time
        profile = config.get('sensor_profile')
        if profile == 'mid360s_raw_lio':
            raise ValueError('Raw MID-360S input was retired. Use mid360s_lio with /bot1/cloud_registered_sparse')
        self.mid360s = profile in ('mid360s_lio', 'mid360s_lio_utlidar')
        self.mixed = MixedCloud(config) if profile == 'mid360s_lio_utlidar' else None
        self.mount = self.mixed.mount if self.mixed else None
        if profile == 'mid360s_lio' and config.get('lio_imu_frame', config['base_frame']) != config['base_frame']:
            self.mount = LioMount(config['mid360_mount'])
        self.imu_samples = deque(maxlen=256)
        rclpy.init()
        self.node = rclpy.create_node(node_name)
        try:
            if self.node.get_parameter('use_sim_time').value:
                raise ValueError('Real-robot sensor input does not support use_sim_time')
            self.timebase = timebase if timebase is not None else self._bootstrap_timebase(
                Odometry, PointCloud2, qos_profile_sensor_data)
        except BaseException:
            try:
                self.node.destroy_node()
            finally:
                rclpy.shutdown()
            raise
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.lock = threading.Lock()
        # Signals each stored LiDAR frame, so acquisition can run on arrival
        # instead of aliasing a fixed poll against the driver's cadence.
        self.frame_ready = threading.Condition(self.lock)
        self.lidar_frames = 0
        self.matched_scan = None
        self.registered_poses = deque(maxlen=8)
        self.registered_scans = deque(maxlen=8)
        self.previous_scan_end_at = None
        self.pose = self.lidar = self.camera = None
        self.images = {}
        self.last_error = None
        self.sequence = 0
        self.pose_history = PoseHistory(config['pose_cloud_tolerance_s'])
        self.last_stamps = {}
        self._errors = {}
        self.continuity = PoseContinuity(config['max_position_jump_m'], config['max_orientation_jump_rad'])
        if self.mid360s:
            from sensor_msgs.msg import Imu
            self.node.create_subscription(Imu, config['imu_topic'], self._guard(self._imu, 'imu_topic'),
                                          qos_profile_sensor_data)
        if self.mixed:
            self.node.create_subscription(Odometry, config['cloud_odom_topic'],
                                          self._guard(self._cloud_odom, 'cloud_odom_topic'), qos_profile_sensor_data)
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

    def _bootstrap_timebase(self, odometry_type, cloud_type, qos):
        """Estimate one shared relative reference before exposing observations.

        This replaces the old sync_time.py receipt estimate, without writing
        CLOCK_REALTIME. A steady common transport delay remains unobservable.
        Never repeat this estimate while a run is active.
        """
        types = {'odom': odometry_type, 'lidar': cloud_type}
        if self.mid360s:
            from sensor_msgs.msg import Imu
            types['imu'] = Imu
        if self.mixed:
            types['cloud_odom'] = odometry_type
        samples = {kind: [] for kind in types}
        floor_scans, positions = [], []
        measure_floor = self.mid360s and self.config['ground_z'] is None
        subscriptions = []
        def receive(kind):
            def callback(message):
                receipt = time.monotonic_ns()
                stamp = message.header.stamp
                if not 0 <= stamp.nanosec < 1_000_000_000:
                    raise ValueError('ROS timestamp invalid during startup')
                seq = stamp.sec*1_000_000_000 + stamp.nanosec
                samples[kind].append((receipt, seq))
                if not measure_floor:
                    return
                if kind == 'odom':
                    self._check_pose_frame(message)
                    p, q = odometry_pose(message)
                    positions.append(p)
                    if self.mixed:
                        self.mixed.lio.add(seq, np.r_[p, np.zeros(3)], q)
                elif kind == 'cloud_odom':
                    self._check_cloud_odom(message)
                    self.mixed.unitree.add(seq, *odometry_pose(message))
                elif kind == 'lidar' and len(floor_scans) < 20:
                    if self.mixed:
                        if message.header.frame_id != self.config['cloud_odom_frame']:
                            raise ValueError('Unexpected Unitree deskewed cloud frame')
                        floor_scans.append((seq, decode_cloud(message, strip_utlidar_padding=True)))
                    else:
                        self._check_registered_frame(message)
                        floor_scans.append(decode_cloud(message))
            return callback
        try:
            for kind, cls in types.items():
                subscriptions.append(self.node.create_subscription(
                    cls, self.config[kind+'_topic'], receive(kind), qos))
            deadline = time.monotonic()+5.
            while time.monotonic() < deadline:
                self.rclpy.spin_once(self.node, timeout_sec=.05)
                odom = samples['odom']
                if len(odom) >= 2 and odom[-1][0]-odom[0][0] >= 3_000_000_000:
                    break
            now_ns = time.monotonic_ns()
            for kind, values in samples.items():
                if self.node.count_publishers(self.config[kind+'_topic']) != 1:
                    raise ValueError(f'{kind}: startup requires exactly one publisher')
                if not values or (now_ns-values[-1][0])*1e-9 > self.config['max_data_age_s']:
                    raise ValueError(f'{kind}: no recent data during time-reference initialization')
            if self.mixed:
                bridge = SourceTimeBridge.from_mixed_samples(samples, max_age_s=self.config['max_data_age_s'])
                aligned = []
                for stamp, points in floor_scans:
                    try:
                        aligned.append(self.mixed.project(stamp, stamp+bridge.unitree_to_lio_ns, points)[3])
                    except ValueError:
                        continue  # startup scans outside either pose history
                floor_scans = aligned
                self.mixed.unitree.samples.clear()
                self.mixed.lio.samples.clear()
            else:
                bridge = SourceTimeBridge.from_samples(samples, max_age_s=self.config['max_data_age_s'],
                                                       clock_reference='imu' if self.mid360s else 'odom')
            if floor_scans and positions:
                self.config['ground_z'] = registered_floor_height(
                    np.concatenate(floor_scans), np.median(positions, axis=0))
            return bridge
        finally:
            for subscription in subscriptions:
                self.node.destroy_subscription(subscription)

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

    def _stamp(self, message, offset_ns=0):
        stamp = message.header.stamp
        if not 0 <= stamp.nanosec < 1_000_000_000:
            raise ValueError('ROS timestamp invalid')
        seq = stamp.sec*1_000_000_000 + stamp.nanosec
        if seq <= 0:
            raise ValueError('ROS timestamp invalid')
        seq += offset_ns
        return seq, self.timebase.captured_at(seq, time.monotonic_ns(), self.config['max_data_age_s'])

    def _tf(self, frame, stamp):
        if not frame:
            raise ValueError('Empty sensor coordinate frame')
        if frame == self.config['map_frame']:
            return np.zeros(3), np.array([1., 0, 0, 0])
        tf = self.buffer.lookup_transform(self.config['map_frame'], frame, self.Time.from_msg(stamp)).transform
        return np.array([tf.translation.x, tf.translation.y, tf.translation.z]), np.array(
            [tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z])

    def _check_pose_frame(self, message):
        child = (self.config.get('lio_imu_frame', self.config['base_frame'])
                 if self.mid360s else self.config['base_frame'])
        if message.child_frame_id != child:
            raise ValueError(f'Unexpected odometry child frame, expected {child}')
        if self.mid360s and message.header.frame_id != self.config['map_frame']:
            raise ValueError('Unexpected Swarm-LIO odometry parent frame')

    def _pose(self, message):
        seq, received = self._stamp(message)
        self._check_pose_frame(message)
        if seq <= self.last_stamps.get('pose', -1):
            raise ValueError('Non-increasing odometry timestamp')
        # Odometry already contains parent->child pose; TF publication is optional.
        p, q = odometry_pose(message)
        if not self.mid360s:
            parent_p, parent_q = self._tf(message.header.frame_id, message.header.stamp)
            p, q = transform(p, parent_p, parent_q), multiply(parent_q, q)
        v = message.twist.twist
        linear = (v.linear.x, v.linear.y, v.linear.z)
        angular = (v.angular.x, v.angular.y, v.angular.z)
        if self.mount is not None and not self.mixed:
            p, q, linear = self.mount.body_state(p, q, linear)
        elif self.mid360s:
            # Swarm-LIO reports world linear velocity. latest() pairs the IMU gyro.
            linear = rotation(q).T @ linear
        velocity = (linear[0], linear[1], angular[2])
        if not np.isfinite((*linear, *angular)).all():
            raise ValueError('Invalid odometry velocity')
        self.continuity.check(p, q)
        with self.lock:
            if self.mixed:
                self.mixed.lio.add(seq, np.r_[p, v.linear.x, v.linear.y, v.linear.z], q)
            elif not self.mid360s:
                self.pose_history.add(seq, p, q)
            self.last_stamps['pose'] = seq
            self.pose = (seq, received, p, q, velocity)
            self.body_twist = (linear, angular)
            if self.mixed:
                self.frame_ready.notify_all()
            elif self.mid360s:
                self.registered_poses.append((self.pose, self.body_twist))
                self._match_scan()

    def _check_cloud_odom(self, message):
        if (message.header.frame_id != self.config['cloud_odom_frame']
                or message.child_frame_id != self.config['base_frame']):
            raise ValueError('Unexpected Unitree scan registration odometry frames')

    def _cloud_odom(self, message):
        self._check_cloud_odom(message)
        offset = self.timebase.unitree_to_lio_ns
        stamp, _ = self._stamp(message, offset)
        p, q = odometry_pose(message)
        with self.frame_ready:
            self.mixed.unitree.add(stamp-offset, p, q)
            if self.mixed.pending:
                self.frame_ready.notify_all()

    def _imu(self, message):
        seq, _ = self._stamp(message)
        if message.header.frame_id != self.config['imu_frame']:
            raise ValueError('Unexpected MID-360S IMU frame')
        gyro, acc = message.angular_velocity, message.linear_acceleration
        angular = (gyro.x, gyro.y, gyro.z)
        # This Livox driver publishes acceleration in g.
        accel = np.array([acc.x, acc.y, acc.z]) * 9.80665
        if not np.isfinite((*angular, *accel)).all():
            raise ValueError('Invalid MID-360S IMU sample')
        with self.lock:
            if self.imu_samples and seq <= self.imu_samples[-1][0]:
                raise ValueError('Non-increasing MID-360S IMU timestamp')
            self.imu_samples.append((seq, angular, accel))

    def _check_registered_frame(self, message):
        if message.header.frame_id != self.config['map_frame']:
            raise ValueError('Expected registered MID-360S points in the LIO world frame')

    def _lidar(self, message):
        if self.mixed:
            offset = self.timebase.unitree_to_lio_ns
            stamp, captured_at = self._stamp(message, offset)
            if message.header.frame_id != self.config['cloud_odom_frame']:
                raise ValueError('Unexpected Unitree deskewed cloud frame')
            if stamp <= self.last_stamps.get('lidar', -1):
                raise ValueError('Non-increasing point-cloud timestamp')
            points = decode_cloud(message, strip_utlidar_padding=True)
            with self.frame_ready:
                self.last_stamps['lidar'] = stamp
                self.mixed.pending.append((stamp-offset, stamp, captured_at, points))
                self.frame_ready.notify_all()
            return
        seq, received = self._stamp(message)
        if self.mid360s:
            self._check_registered_frame(message)
        if seq <= self.last_stamps.get('lidar', -1):
            raise ValueError('Non-increasing point-cloud timestamp')
        points = decode_cloud(message, strip_utlidar_padding=(
            self.config['lidar_topic'] == '/utlidar/cloud_deskewed'
            and message.header.frame_id == 'odom'))
        if not self.mid360s:
            p, q = self._tf(message.header.frame_id, message.header.stamp)
            points = transform(points, p, q).astype(np.float32)
        lidar = {'pointcloud': points, 'physics_step': seq, 'max_range': self.config['lidar_max_range']}
        with self.lock:
            self.last_stamps['lidar'] = seq
            if self.mid360s:
                # The header identifies the scan END. The preceding scan end
                # bounds its oldest returns, including after a dropped message.
                start_at = self.previous_scan_end_at
                self.previous_scan_end_at = received
                if start_at is None:
                    return
                lidar['source_stamp_ns'] = seq
                self.lidar = (start_at, lidar)
                self.registered_scans.append(self.lidar)
                self._match_scan()
            else:
                self.lidar = (received, lidar)
                self.lidar_frames += 1
                self.frame_ready.notify_all()

    def _match_scan(self):
        # Each DDS stream is ordered, but one can arrive several scans ahead.
        while self.registered_poses and self.registered_scans:
            pose, twist = self.registered_poses[0]
            scan = self.registered_scans[0]
            stamp = scan[1]['physics_step']
            if pose[0] == stamp:
                self.matched_scan = (pose, scan, twist)
                self.frame_ready.notify_all()
            if pose[0] <= stamp:
                self.registered_poses.popleft()
            if stamp <= pose[0]:
                self.registered_scans.popleft()

    def wait_frame(self, seen, timeout):
        """Block until a LiDAR frame other than count ``seen`` is stored; return the count."""
        with self.frame_ready:
            def frame_id():
                if self.mixed:
                    pair = self.mixed.peek()
                    if pair is not None:
                        return pair[0][0]
                if self.mid360s:
                    return self.matched_scan[0][0] if self.matched_scan else None
                return self.lidar_frames
            self.frame_ready.wait_for(lambda: frame_id() != seen, timeout=max(0., timeout))
            return frame_id()

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
            if self.pose is None or (not self.mixed and self.lidar is None):
                return None
            if self.mid360s:
                if self.mixed:
                    self.matched_scan = self.mixed.take() or self.matched_scan
                if self.matched_scan is None:
                    self.last_error = 'Waiting for time-aligned odometry and scan'
                    return None
                pose, scan, twist = self.matched_scan
                cloud_p, cloud_q = pose[2:4]
            else:
                pose, scan, twist = self.pose, self.lidar, getattr(self, 'body_twist', None)
                try:
                    cloud_p, cloud_q = self.pose_history.at(scan[1]['physics_step'])
                except ValueError as error:
                    self.last_error = str(error)
                    return None
            seq, pose_at, p, q, velocity = pose
            lidar_at, lidar = scan
            lidar = dict(lidar)
            self._pair_error = None
            offset = self.config['lidar_origin_offset_base_m']
            calibrated = offset is not None and self.config['ground_z'] is not None
            # Missing calibration permits recording only; runtime must not build a map.
            lidar['position'] = transform(offset if offset is not None else [0, 0, 0], cloud_p, cloud_q)
            sensors = {'lidar': lidar}
            if self.mid360s:
                imu = min(self.imu_samples, key=lambda v: abs(v[0]-seq), default=None)
                if imu is None or abs(imu[0]-seq) > self.pose_history.tolerance_ns:
                    self.last_error = 'Waiting for MID-360S IMU at odometry time'
                    return None
                stamp, angular, accel = imu
                linear = twist[0]
                sensors['imu'] = dict(source_stamp_ns=stamp, frame_id=self.config['imu_frame'],
                                      angular_velocity_rad_s=angular, linear_acceleration_m_s2=accel)
                if self.mount is not None:
                    angular = self.mount.body_from_imu @ angular
                    linear = linear - np.cross(angular, self.mount.imu_in_body)
                twist = (linear, angular)
                velocity = (linear[0], linear[1], angular[2])
            camera_at = None
            if self.camera is not None:
                camera_at, sensors['camera'] = self.camera
            raw = {'position': p, 'orientation': q, 'sensors': sensors}
            if twist is not None:
                raw.update(body_linear_velocity=twist[0], body_angular_velocity=twist[1])
            observation = normalize(raw, self.config['ground_z'] if calibrated else 0.)
        self.sequence += 1
        return Snapshot(observation, time.monotonic(), pose_at, lidar_at, camera_at, velocity, self.sequence,
                        geometry_calibrated=calibrated, planning_frame=self.config['map_frame'])

    def close(self):
        self.rclpy.shutdown()
        self.thread.join(timeout=2)
        self.node.destroy_node()
