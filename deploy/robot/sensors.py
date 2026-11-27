"""ROS 2 inputs for registered MID-360S scans, Swarm-LIO odometry and Livox IMU."""
from collections import deque
import threading
import time
import numpy as np
from .localization import LioMount, PoseContinuity, odometry_pose, registered_floor_height, rotation, transform
from .observation import Snapshot, normalize
from .timebase import SourceTimeBridge


def decode_image(message):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(message.encoding)
    if channels is None:
        raise ValueError(f'Unsupported RGB preview encoding {message.encoding}')
    row_bytes = message.width * channels
    if message.step < row_bytes or len(message.data) < message.height * message.step:
        raise ValueError('Invalid image stride or payload')
    image = np.ndarray((message.height, message.width, channels), dtype=np.uint8,
                       buffer=bytes(message.data), strides=(message.step, channels, 1)).copy()
    if message.encoding in ('bgr8', 'bgra8'):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return image


def decode_cloud(message):
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
        from sensor_msgs.msg import PointCloud2, Image, Imu
        from nav_msgs.msg import Odometry
        self.config, self.rclpy = dict(config), rclpy
        self.mount = None
        if config.get('lio_imu_frame', config['base_frame']) != config['base_frame']:
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
        self.lock = threading.Lock()
        # Signals each stored LiDAR frame, so acquisition can run on arrival
        # instead of aliasing a fixed poll against the driver's cadence.
        self.frame_ready = threading.Condition(self.lock)
        self.matched_scan = None
        self.registered_poses = deque(maxlen=8)
        self.registered_scans = deque(maxlen=8)
        self.previous_scan_end_at = None
        self.last_error = None
        self.sequence = 0
        self.imu_tolerance_ns = round(config['pose_cloud_tolerance_s'] * 1e9)
        self.last_stamps = {}
        self._errors = {}
        self.continuity = PoseContinuity(config['max_position_jump_m'], config['max_orientation_jump_rad'])
        for topic, cls, callback in (
            ('odom_topic', Odometry, self._pose),
            ('lidar_topic', PointCloud2, self._lidar),
            ('imu_topic', Imu, self._imu),
        ):
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
        from sensor_msgs.msg import Imu
        types = {'odom': odometry_type, 'lidar': cloud_type, 'imu': Imu}
        samples = {kind: [] for kind in types}
        floor_scans, positions = [], []
        measure_floor = self.config['ground_z'] is None
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
                    position, _ = odometry_pose(message)
                    positions.append(position)
                elif kind == 'lidar' and len(floor_scans) < 20:
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
            bridge = SourceTimeBridge.from_samples(samples, max_age_s=self.config['max_data_age_s'],
                                                   clock_reference='imu')
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

    def _stamp(self, message):
        stamp = message.header.stamp
        if not 0 <= stamp.nanosec < 1_000_000_000:
            raise ValueError('ROS timestamp invalid')
        seq = stamp.sec*1_000_000_000 + stamp.nanosec
        if seq <= 0:
            raise ValueError('ROS timestamp invalid')
        return seq, self.timebase.captured_at(seq, time.monotonic_ns(), self.config['max_data_age_s'])

    def _check_pose_frame(self, message):
        child = self.config.get('lio_imu_frame', self.config['base_frame'])
        if message.child_frame_id != child:
            raise ValueError(f'Unexpected odometry child frame, expected {child}')
        if message.header.frame_id != self.config['map_frame']:
            raise ValueError('Unexpected Swarm-LIO odometry parent frame')

    def _pose(self, message):
        seq, received = self._stamp(message)
        self._check_pose_frame(message)
        if seq <= self.last_stamps.get('pose', -1):
            raise ValueError('Non-increasing odometry timestamp')
        # Odometry already contains parent->child pose; TF publication is optional.
        p, q = odometry_pose(message)
        v = message.twist.twist
        linear = (v.linear.x, v.linear.y, v.linear.z)
        angular = (v.angular.x, v.angular.y, v.angular.z)
        if self.mount is not None:
            p, q, linear = self.mount.body_state(p, q, linear)
        else:
            # Swarm-LIO reports world linear velocity. latest() pairs the IMU gyro.
            linear = rotation(q).T @ linear
        if not np.isfinite((*linear, *angular)).all():
            raise ValueError('Invalid odometry velocity')
        self.continuity.check(p, q)
        with self.lock:
            self.last_stamps['pose'] = seq
            pose = (seq, received, p, q)
            self.registered_poses.append((pose, linear))
            self._match_scan()

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
        seq, received = self._stamp(message)
        self._check_registered_frame(message)
        if seq <= self.last_stamps.get('lidar', -1):
            raise ValueError('Non-increasing point-cloud timestamp')
        points = decode_cloud(message)
        lidar = {'pointcloud': points, 'physics_step': seq, 'source_stamp_ns': seq,
                 'max_range': self.config['lidar_max_range']}
        with self.lock:
            self.last_stamps['lidar'] = seq
            # The previous scan end bounds the oldest returns, including after a gap.
            start_at = self.previous_scan_end_at
            self.previous_scan_end_at = received
            if start_at is not None:
                self.registered_scans.append((start_at, lidar))
                self._match_scan()

    def _match_scan(self):
        # Each DDS stream is ordered, but one can arrive several scans ahead.
        while self.registered_poses and self.registered_scans:
            pose, linear = self.registered_poses[0]
            scan = self.registered_scans[0]
            stamp = scan[1]['physics_step']
            if pose[0] == stamp:
                self.matched_scan = (pose, scan, linear)
                self.frame_ready.notify_all()
            if pose[0] <= stamp:
                self.registered_poses.popleft()
            if stamp <= pose[0]:
                self.registered_scans.popleft()

    def wait_frame(self, seen, timeout):
        """Block until a paired scan has a different source stamp, or the timeout elapses."""
        with self.frame_ready:
            def frame_id():
                return self.matched_scan[0][0] if self.matched_scan else None
            self.frame_ready.wait_for(lambda: frame_id() != seen, timeout=max(0., timeout))
            return frame_id()

    def latest(self):
        if self.continuity.fault:
            raise ValueError(self.continuity.fault)
        with self.lock:
            if self.matched_scan is None:
                self.last_error = 'Waiting for time-aligned odometry and scan'
                return None
            pose, scan, linear = self.matched_scan
            seq, pose_at, p, q = pose
            lidar_at, lidar = scan
            lidar = dict(lidar)
            self._pair_error = None
            offset = self.config['lidar_origin_offset_base_m']
            calibrated = offset is not None and self.config['ground_z'] is not None
            # Missing calibration permits recording only; runtime must not build a map.
            lidar['position'] = transform(offset if offset is not None else [0, 0, 0], p, q)
            sensors = {'lidar': lidar}
            imu = min(self.imu_samples, key=lambda v: abs(v[0]-seq), default=None)
            if imu is None or abs(imu[0]-seq) > self.imu_tolerance_ns:
                self.last_error = 'Waiting for MID-360S IMU at odometry time'
                return None
            stamp, angular, accel = imu
            sensors['imu'] = dict(source_stamp_ns=stamp, frame_id=self.config['imu_frame'],
                                  angular_velocity_rad_s=angular, linear_acceleration_m_s2=accel)
            if self.mount is not None:
                angular = self.mount.body_from_imu @ angular
                linear = linear - np.cross(angular, self.mount.imu_in_body)
            velocity = (linear[0], linear[1], angular[2])
            raw = {'position': p, 'orientation': q, 'sensors': sensors,
                   'body_linear_velocity': linear, 'body_angular_velocity': angular}
            observation = normalize(raw, self.config['ground_z'] if calibrated else 0.)
        self.sequence += 1
        return Snapshot(observation, time.monotonic(), pose_at, lidar_at, None, velocity, self.sequence,
                        geometry_calibrated=calibrated, planning_frame=self.config['map_frame'])

    def close(self):
        self.rclpy.shutdown()
        self.thread.join(timeout=2)
        self.node.destroy_node()
