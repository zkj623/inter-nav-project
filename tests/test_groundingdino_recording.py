import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grutopia_extension.interactive_navigation.visualization import InteractionVideoRecorder


class _Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        pass


class GroundingDinoRecordingTest(unittest.TestCase):
    def test_detector_video_and_jsonl_use_exact_query_frame_without_isaac_boxes(self):
        writers = {}

        def make_writer(_recorder, filename, _size):
            writer = _Writer()
            writers[filename] = writer
            return writer

        debug_frame = {
            'step': 24,
            'status': 'ok',
            'query': 'refrigerator . door .',
            'image_size': [80, 60],
            'boxes': [{
                'label': 'refrigerator',
                'confidence': 0.8,
                'bbox_xyxy': [10.0, 10.0, 20.0, 20.0],
                'above_threshold': True,
            }],
            'mapped': [],
            'item_errors': [],
        }
        camera = {
            'rgba': np.zeros((60, 80, 4), dtype=np.uint8),
            'bounding_box_2d_tight': {
                'data': [(1, 40, 30, 50, 40)],
                'info': {'idToLabels': {'1': {'class': 'door'}}},
            },
        }
        observation = {'sensors': {'camera': camera}}
        runtime = SimpleNamespace(
            semantic_detection_mode=SimpleNamespace(uses_open_vocabulary=True),
            open_vocabulary_perception=SimpleNamespace(last_debug_frame=debug_frame),
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(InteractionVideoRecorder, '_writer', make_writer):
                recorder = InteractionVideoRecorder(directory, capture_interval=1)
                recorder.record_perception(24, observation, runtime)
                recorder.record_perception(24, observation, runtime)
                recorder.record_perception(25, observation, runtime)
                with patch.object(
                    recorder, '_render_map', return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
                ), patch.object(
                    recorder, '_render_pointcloud', return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
                ):
                    position = SimpleNamespace(robot_position=(0.0, 0.0, 0.0))
                    recorder.capture(24, 'exploring', observation, position, runtime)
                    recorder.capture(25, 'exploring', observation, position, runtime)
                recorder.close()

            events = (Path(directory) / 'groundingdino_detections.jsonl').read_text().splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0]), {**debug_frame, 'video_frame_index': 0})
            video_frames = [
                json.loads(line) for line in
                (Path(directory) / 'video_frames.jsonl').read_text().splitlines()
            ]
            self.assertEqual([frame['step'] for frame in video_frames], [24, 25])
            self.assertEqual([frame['frame_index'] for frame in video_frames], [0, 1])
            self.assertEqual(len(writers['groundingdino.mp4'].frames), 1)
            frame = writers['groundingdino.mp4'].frames[0]
            self.assertEqual(tuple(frame[60, 80]), (220, 60, 230))
            self.assertEqual(tuple(frame[180, 320]), (0, 0, 0))
            first, second = writers['robot_rgb.mp4'].frames
            self.assertEqual(tuple(first[60, 80]), (220, 60, 230))
            self.assertEqual(tuple(first[180, 320]), (0, 0, 0))
            self.assertEqual(tuple(second[60, 80]), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
