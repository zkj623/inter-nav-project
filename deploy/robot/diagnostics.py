"""Bounded, subscription-only ROS inspection. Never imports the motion backend."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


AUDITED_TOPICS = {
    '/utlidar/cloud': 'sensor_msgs/msg/PointCloud2',
    '/utlidar/cloud_base': 'sensor_msgs/msg/PointCloud2',
    '/utlidar/cloud_deskewed': 'sensor_msgs/msg/PointCloud2',
    '/utlidar/imu': 'sensor_msgs/msg/Imu',
    '/utlidar/robot_odom': 'nav_msgs/msg/Odometry',
    '/utlidar/robot_pose': 'geometry_msgs/msg/PoseStamped',
    '/frontvideostream': 'unitree_go/msg/Go2FrontVideoData',
    '/uslam/frontend/odom': 'nav_msgs/msg/Odometry',
    '/uslam/localization/odom': 'nav_msgs/msg/Odometry',
    '/lio_sam_ros2/mapping/odometry': 'nav_msgs/msg/Odometry',
}


def describe(message, ros_now):
    """Keep metadata only: never serialize an entire video or point-cloud payload."""
    result = {}
    if hasattr(message, 'header'):
        header = message.header
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        result.update(frame_id=header.frame_id, stamp_s=stamp, age_s=ros_now-stamp)
    if hasattr(message, 'child_frame_id'):
        result['child_frame_id'] = message.child_frame_id
        p = message.pose.pose.position
        result['position'] = [p.x, p.y, p.z]
    if hasattr(message, 'point_step'):
        result.update(width=message.width, height=message.height,
                      point_step=message.point_step, row_step=message.row_step,
                      payload_bytes=len(message.data),
                      expected_bytes=message.height * message.row_step,
                      fields=[{'name': f.name, 'offset': f.offset, 'datatype': f.datatype,
                               'count': f.count} for f in message.fields])
    elif hasattr(message, 'data'):
        try:
            result['data_elements'] = len(message.data)
        except TypeError:
            pass
    return result


def run(config, duration, output):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message
    from ament_index_python.packages import get_package_share_directory

    robot = config['robot']
    topics = dict(AUDITED_TOPICS)
    topics.update({robot['odom_topic']: 'nav_msgs/msg/Odometry',
                   robot['lidar_topic']: 'sensor_msgs/msg/PointCloud2',
                   robot['front_video_topic']: 'unitree_go/msg/Go2FrontVideoData'})
    if robot['camera_mode'] == 'rgbd':
        topics.update({robot['rgb_topic']: 'sensor_msgs/msg/Image',
                       robot['depth_topic']: 'sensor_msgs/msg/Image',
                       robot['camera_info_topic']: 'sensor_msgs/msg/CameraInfo'})
    report = {
        'read_only': True, 'hardware_validated': False, 'duration_s': duration,
        'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'architecture': platform.machine(),
        'environment': {k: os.environ.get(k) for k in (
            'ROS_DISTRO', 'ROS_VERSION', 'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION',
            'CYCLONEDDS_URI', 'AMENT_PREFIX_PATH', 'PYTHONPATH')},
        'topics': {}, 'local_interfaces': {},
        'limitations': ['Receipt is not global localization or motion acceptance.',
                       'DDS errors before deserialization may never reach callbacks. Save stderr separately.',
                       'Local interface hashes do not establish publisher compatibility; compare on publisher host.'],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('inter_nav_deploy_diagnostics')
    subscriptions = []

    def callback(topic):
        def receive(message):
            state = report['topics'][topic]
            now = time.monotonic()
            state['received'] += 1
            state.setdefault('first_received_monotonic', now)
            previous = state.get('last_received_monotonic')
            if previous is not None:
                state['max_gap_s'] = max(state.get('max_gap_s', 0), now-previous)
            state['last_received_monotonic'] = now
            try:
                sample = describe(message, node.get_clock().now().nanoseconds*1e-9)
                previous_sample = state.get('last_sample', {})
                if 'stamp_s' in sample:
                    for name, failed in (
                        ('non_increasing_stamps', sample['stamp_s'] <= previous_sample.get('stamp_s', -1)),
                        ('stale_messages', sample['age_s'] > robot['max_data_age_s']),
                        ('future_messages', sample['age_s'] < -.05),
                    ):
                        state[name] = state.get(name, 0)+int(failed)
                state['last_sample'] = sample
            except Exception as error:
                state['sample_error'] = str(error)
        return receive

    try:
        for topic, type_name in topics.items():
            state = report['topics'][topic] = {'expected_type': type_name, 'received': 0}
            try:
                cls = get_message(type_name)
                state['local_fields'] = cls.get_fields_and_field_types()
                package, category, name = type_name.split('/')
                share = Path(get_package_share_directory(package))
                for suffix in ('.msg', '.idl'):
                    source = share / category / (name+suffix)
                    if source.is_file():
                        report['local_interfaces'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                subscriptions.append(node.create_subscription(
                    cls, topic, callback(topic), qos_profile_sensor_data))
            except Exception as error:
                state['subscription_error'] = f'{type(error).__name__}: {error}'
        deadline = time.monotonic()+duration
        print(f'只读订阅诊断 {duration:g} 秒；不启动运动控制。', flush=True)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(.1, max(0., deadline-time.monotonic())))
        discovered = dict(node.get_topic_names_and_types())
        for topic, state in report['topics'].items():
            state['discovered_types'] = discovered.get(topic, [])
            state['publishers'] = node.count_publishers(topic)
            if state['received'] > 1:
                elapsed = state['last_received_monotonic']-state['first_received_monotonic']
                state['received_hz'] = (state['received']-1)/elapsed if elapsed > 0 else None
            print(f"{topic}: publishers={state['publishers']} received={state['received']}", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'诊断记录：{destination.resolve()}', flush=True)
    # This indicates input receipt only. DDS compatibility must be assessed separately.
    return 0 if all(report['topics'][robot[key]]['received'] > 0
                    for key in ('odom_topic', 'lidar_topic')) else 2
