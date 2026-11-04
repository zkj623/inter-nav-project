"""Subscription-only control evidence. No SDK clients, publishers, or motion imports."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
import uuid

from deploy.app.config import REPO
from .recorder import json_value, provenance


def control_topics(robot):
    # SDK rt/foo maps to ROS /foo; honour the reviewed robot configuration.
    def ros_topic(name):
        return '/' + (name[3:] if name.startswith('rt/') else name).lstrip('/')
    topics = {
        robot['odom_topic']: ('nav_msgs/msg/Odometry', 100.),
        ros_topic(robot['sport_state_topic']): ('unitree_go/msg/SportModeState', 100.),
        ros_topic(robot['low_state_topic']): ('unitree_go/msg/LowState', 100.),
        '/wirelesscontroller': ('unitree_go/msg/WirelessController', 0.),
        '/wirelesscontroller_unprocessed': ('unitree_go/msg/WirelessController', 0.),
    }
    topics.setdefault('/lf/sportmodestate', ('unitree_go/msg/SportModeState', 0.))
    for service in ('sport', 'obstacles_avoid', 'motion_switcher', 'robot_state'):
        for direction, message in (('request', 'Request'), ('response', 'Response')):
            topics[f'/api/{service}/{direction}'] = (f'unitree_api/msg/{message}', 0.)
    return topics


def message_data(message, convert):
    data = convert(message)
    if 'motor_state' in data:
        # Keep the twelve Go2 leg joints, contact readings, IMU, raw remote,
        # firmware identifier fields and electrical/thermal feedback.
        fields = ('mode', 'q', 'dq', 'ddq', 'tau_est', 'temperature', 'lost')
        data['motor_state'] = [{k: m[k] for k in fields if k in m} for m in data['motor_state'][:12]]
        remote = bytes(data['wireless_remote'])
        data['remote_keys'] = int.from_bytes(remote[2:4], 'little') if len(remote) >= 4 else None
    if 'parameter' in data or ('header' in data and 'data' in data):
        field = 'parameter' if 'parameter' in data else 'data'
        try:
            data[field+'_json'] = json.loads(data[field])
        except (ValueError, TypeError):
            pass  # Retain the original wire string even if it is not JSON.
    return data


class ControlLog:
    """One callback thread; explicit sampling counts, no unbounded queues."""
    def __init__(self, stream, topics, started_ns):
        self.stream, self.started_ns = stream, started_ns
        self.last_written, self.last_keys = {}, {}
        self.stats = {topic: {'type': kind, 'max_record_hz': hz, 'received': 0, 'written': 0,
                             'sampled_out': 0, 'decode_errors': 0, 'max_callback_gap_s': 0.}
                      for topic, (kind, hz) in topics.items()}

    def receive(self, topic, message, convert, received_ns):
        stats = self.stats[topic]
        now = received_ns/1e9
        if 'last_received_monotonic_s' in stats:
            stats['max_callback_gap_s'] = max(stats['max_callback_gap_s'], now-stats['last_received_monotonic_s'])
        stats.setdefault('first_received_monotonic_s', now)
        stats['last_received_monotonic_s'] = now
        stats['received'] += 1
        # Preserve a short button edge even between sampled LowState packets.
        keys = bytes(message.wireless_remote)[2:4] if hasattr(message, 'wireless_remote') else None
        changed = keys is not None and keys != self.last_keys.get(topic)
        self.last_keys[topic] = keys
        hz = stats['max_record_hz']
        if hz and not changed and now-self.last_written.get(topic, float('-inf')) < 1/hz:
            stats['sampled_out'] += 1
            return
        try:
            row = {'topic': topic, 'received_monotonic_ns': received_ns,
                   'elapsed_s': (received_ns-self.started_ns)/1e9, 'data': message_data(message, convert)}
            line = json.dumps(row, default=json_value, allow_nan=False, separators=(',', ':'))
        except (ValueError, TypeError, OverflowError) as exc:
            stats['decode_errors'] += 1
            stats['last_decode_error'] = str(exc)
            return
        self.stream.write(line+'\n')  # Disk errors propagate and terminate recording.
        self.last_written[topic] = now
        stats['written'] += 1


def record_control(config, output=None, duration=60.):
    root = Path(output) if output is not None else Path(config['runtime']['recording']['root'])/'control'
    if not root.is_absolute():
        root = REPO/root
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
    directory = root/name
    directory.mkdir(parents=True, exist_ok=False)
    started_ns = time.monotonic_ns()
    topics = control_topics(config['robot'])
    required = [topic for topic in topics if topics[topic][1] == 100.]
    remote_topics = ['/wirelesscontroller', '/wirelesscontroller_unprocessed']
    info = {'mode': 'control_trace_read_only', 'schema_version': 1, 'config': config,
            'started_at': datetime.now(timezone.utc).isoformat(), 'started_monotonic_ns': started_ns,
            'duration_limit_s': duration, 'provenance': provenance(REPO),
            'topics': topics, 'required_feedback_topics': required,
            'limitations': ['Only subscribes; no movement, stop or mode-switch requests are sent.',
                           'Feedback is sampled at up to 100 Hz; remote and RPC callbacks are not sampled.',
                           'All times share this host monotonic clock; message/source stamps are preserved unchanged.',
                           'BEST_EFFORT transport loss before a callback cannot be counted here.',
                           'Foxy callbacks do not expose per-message publisher GIDs; endpoint lists are snapshots only.',
                           'No RPC traffic may mean a quiet topic or an internal firmware path, not absence of control.',
                           'Separate wireless topics may be silent; LowState raw wireless_remote bytes are always retained.',
                           'This trace has no point cloud/images and is not a navigation replay recording.']}
    (directory/'session.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f'只读控制录制 → {directory}\n直接开始订阅，不等待 run.py；不发送运动、停止或模式切换指令。', flush=True)
    stream = (directory/'control.jsonl').open('w')
    log = ControlLog(stream, topics, started_ns)
    node = None
    ros = None
    handlers = {}
    stopping = False
    reason, error = 'duration_elapsed', None
    endpoints = {}
    saved_stderr = os.dup(2)
    stderr_log = (directory/'stderr.log').open('w')

    def stop(signum, _):
        nonlocal stopping, reason
        stopping, reason = True, f'signal_{signum}'

    def inspect_publishers():
        return {topic: [{'name': p.node_name, 'namespace': p.node_namespace, 'gid': list(p.endpoint_gid)}
                        for p in node.get_publishers_info_by_topic(topic)] for topic in topics}

    try:
        os.dup2(stderr_log.fileno(), 2)
        import rclpy as ros
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rosidl_runtime_py.utilities import get_message
        from rosidl_runtime_py.convert import message_to_ordereddict
        ros.init()
        node = ros.create_node('inter_nav_control_record_'+uuid.uuid4().hex[:8])
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        def callback(topic):
            def receive(message):
                log.receive(topic, message, message_to_ordereddict, time.monotonic_ns())
            return receive
        subscriptions = [node.create_subscription(get_message(kind), topic, callback(topic), qos)
                         for topic, (kind, _) in topics.items()]
        deadline = time.monotonic()+duration
        next_status = time.monotonic()+2
        ready = False
        while not stopping and ros.ok() and time.monotonic() < deadline:
            ros.spin_once(node, timeout_sec=.05)
            now = time.monotonic()
            if now >= next_status:
                stream.flush()
                next_status = now+2
                missing = [topic for topic in required if not log.stats[topic]['written']
                           or now-log.stats[topic]['last_received_monotonic_s'] > .5]
                missing_remote = [topic for topic in remote_topics if not log.stats[topic]['written']
                                  or now-log.stats[topic]['last_received_monotonic_s'] > .5]
                if not missing and not ready:
                    print('状态、足部和里程计已收到；LowState 原始遥控字节随包保存。可以开始本轮遥控实验。', flush=True)
                    ready = True
                    endpoints['after_discovery'] = inspect_publishers()
                print(f'已记录 {(time.monotonic_ns()-started_ns)/1e9:.1f} 秒；'
                      f'消息 {sum(s["written"] for s in log.stats.values())}；缺少或中断反馈 {missing}；'
                      f'独立摇杆话题缺失或中断 {missing_remote}', flush=True)
        if not stopping and not ros.ok():
            reason = 'ros_shutdown'
        endpoints['at_close'] = inspect_publishers()
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except Exception as exc:
        error, reason = f'{type(exc).__name__}: {exc}', 'record_error'
        print(f'控制录制失败：{error}', flush=True)
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if ros is not None and ros.ok():
                ros.shutdown()
        except Exception as exc:
            error = error or f'cleanup: {type(exc).__name__}: {exc}'
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            stderr_log.close()
            stream.close()
            missing = [topic for topic in required if not log.stats[topic]['written']]
            summary = {'reason': reason, 'error': error, 'ended_monotonic_ns': time.monotonic_ns(),
                       'finished_cleanly': error is None, 'missing_feedback_topics': missing,
                       'missing_remote_topics': [t for t in remote_topics if not log.stats[t]['written']],
                       'topics': log.stats, 'publishers': endpoints,
                       'transport_completeness_verified': False, 'hardware_validated': False}
            (directory/'control_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f'录制结束：{directory}\n只停止了录制；机器人仍由现场遥控器控制。', flush=True)
    return 2 if error or missing or any(s['decode_errors'] for s in log.stats.values()) else 0
