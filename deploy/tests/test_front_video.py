"""Wire-layout regression checks without ROS, SDK, camera or native allocation."""
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from deploy.robot.front_video import describe_front_video, MAX_SERIALIZED_BYTES
from deploy.robot import diagnostics


def encapsulate(body, endian='<', padded=True):
    pad = (-len(body)) % 4 if padded else 0
    return (b'\x00\x01' if endian == '<' else b'\x00\x00') + bytes([0, pad]) + body + b'\0'*pad


def tagged(tag=720, payload=b'\x00\x00\x00\x01\x41abc', endian='<'):
    return encapsulate(struct.pack(endian+'QII', 445539729, tag, len(payload))+payload, endian)


class FrontVideoTests(unittest.TestCase):
    def test_observed_layout_endianness_and_all_padding_sizes(self):
        for endian in ('<', '>'):
            for tag in (180, 360, 720):
                for suffix in range(4):
                    payload = b'\x00\x00\x00\x01\x41secret'+b'x'*suffix
                    result = describe_front_video(tagged(tag, payload, endian))
                    self.assertEqual(result['wire_layout'], 'tagged_single_stream')
                    self.assertFalse(result['declared_schema_matches'])
                    self.assertEqual(result['stream_tag'], tag)
                    self.assertEqual(result['encoded_bytes'], len(payload))
                    self.assertNotIn('secret', json.dumps(result))

    def test_legacy_arrays_with_alignment_and_empty_sequences(self):
        for endian in ('<', '>'):
            for arrays in ((b'a', b'bc', b'def'), (b'', b'', b'')):
                body = struct.pack(endian+'Q', 7)
                for array in arrays:
                    body += b'\0'*((-len(body)) % 4)
                    body += struct.pack(endian+'I', len(array))+array
                result = describe_front_video(encapsulate(body, endian))
                self.assertTrue(result['declared_schema_matches'])
                self.assertEqual(list(result['encoded_bytes'].values()), list(map(len, arrays)))

    def test_malformed_lengths_headers_and_unknown_layouts_rejected(self):
        good = tagged()
        bad_length = bytearray(good)
        struct.pack_into('<I', bad_length, 16, 0xffffffff)
        cases = [b'', good[:10], good[:-5], good+b'junk', bytes(bad_length),
                 b'\x00\x07'+good[2:], good[:2]+b'\x01\x00'+good[4:],
                 tagged(999), tagged(payload=b'not a video'),
                 b'\0'*(MAX_SERIALIZED_BYTES+1)]
        for data in cases:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                describe_front_video(data)

    def test_diagnostic_receives_raw_video_and_reports_bad_envelopes(self):
        for payload, expected_exit in ((tagged(), 0), (b'bad packet', 2)):
            with self.subTest(expected_exit=expected_exit), TemporaryDirectory() as directory:
                subscriptions = []
                node = Mock()
                def subscribe(cls, topic, callback, qos, raw=False):
                    subscriptions.append((topic, callback, raw))
                    return object()
                node.create_subscription.side_effect = subscribe
                node.count_publishers.return_value = 1
                node.get_clock.return_value.now.return_value.nanoseconds = 1000000000
                node.get_topic_names_and_types.return_value = [
                    (topic, [kind]) for topic, kind in diagnostics.AUDITED_TOPICS.items()]
                def spin(node, timeout_sec):
                    for topic, callback, raw in subscriptions:
                        callback(payload if raw else SimpleNamespace())
                ros = SimpleNamespace(init=lambda: None, shutdown=lambda: None,
                                      create_node=lambda name: node, spin_once=spin)
                cls = SimpleNamespace(get_fields_and_field_types=lambda: {})
                modules = {
                    'rclpy': ros,
                    'rclpy.qos': SimpleNamespace(qos_profile_sensor_data=object()),
                    'rosidl_runtime_py.utilities': SimpleNamespace(get_message=lambda name: cls),
                    'ament_index_python.packages': SimpleNamespace(get_package_share_directory=lambda name: directory),
                }
                ticks = iter(i*.1 for i in range(1000))
                config = {'robot': {'odom_topic': '/utlidar/robot_odom',
                                   'lidar_topic': '/utlidar/cloud_deskewed',
                                   'front_video_topic': '/frontvideostream',
                                   'camera_mode': 'unavailable', 'max_data_age_s': .4}}
                output = Path(directory) / 'report.json'
                with patch.dict('sys.modules', modules), patch.object(
                        diagnostics.time, 'monotonic', side_effect=lambda: next(ticks)), patch('builtins.print'):
                    self.assertEqual(diagnostics.run(config, 1, output), expected_exit)
                self.assertEqual([topic for topic, _, raw in subscriptions if raw], ['/frontvideostream'])
                result = json.loads(output.read_text())
                self.assertFalse(result['hardware_validated'])
                video = result['topics']['/frontvideostream']
                self.assertEqual(video['received'], 1)
                self.assertEqual(video['validated_messages'], int(expected_exit == 0))
                self.assertEqual(video.get('sample_error_count', 0), int(expected_exit != 0))
