"""Online map snapshots and interaction diagnostics remain read-only."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from deploy.robot.observation import Snapshot
from deploy.app.session import encode_packet, decode_packet
from deploy.visualization.scene import SceneCapture, map_payload, render_map
from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision, BoxPushPlanner, BoxState
from grutopia_extension.interactive_navigation.mapping import MappingConfig
from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap


def test_online_layers_and_push_state_are_detached_from_display():
    config = MappingConfig(x_limits=(-2., 2.), y_limits=(-2., 2.), grid_resolution=.1)
    planner = BoxPushPlanner(np.zeros((40, 40), dtype=bool),
                             (config.x_limits, config.y_limits), resolution=.1)
    layers = OnlineNavigationMap(planner, config)
    box = BoxState('observed_box_7', (.8, .6), (.4, .3, .5))
    layers.current_boxes = [box]
    layers.observed[15:25, 15:25] = True
    layers.tracked_occupied[25:28, 27:31] = True
    layers.local_revision, layers.global_revision = 4, 2
    decision = BoxDecision('push', 'shorter_observed_route', explanation={
        'before_length_m': float('inf'), 'after_length_m': 2.5})
    skill = SimpleNamespace(state='settle_contact', status='running', reason='',
                            plan=SimpleNamespace(box_id=box.id))
    engine = SimpleNamespace(layers=layers, planner=planner, goal=(1.5, 0.),
                              path=[(.4, 0., .3)], phase='skill', decision=decision, skill=skill)
    tasks = SimpleNamespace(map_ready=True, engine=engine, last_mapping_at=3.,
                             state='active', mode='navigate', result=None, last_timings={})
    observation = {'position': np.array([0., 0., .3]),
                   'orientation': np.array([1., 0., 0., 0.]), 'sensors': {}}
    snapshot = Snapshot(observation, 3., 3., 3., None, (0., 0., 0.), 1,
                        planning_frame='experiment')
    scene = SceneCapture().capture(snapshot, tasks)
    assert 'points' not in scene
    assert scene['decision']['explanation']['before_length_m'] is None
    assert scene['skill']['state'] == 'settle_contact'
    assert scene['objects'][0]['label'] == box.id
    assert layers.local_revision == 4
    assert decision.explanation['before_length_m'] == float('inf')
    layers.observed[:] = False
    layers.tracked_occupied[:] = False
    engine.path[0] = (9., 9., .3)
    assert scene['observed'].any() and scene['tracked'].any()
    assert scene['path'][0] == [.4, 0., .3]
    assert render_map(scene).shape == (720, 1280, 3)
    scene['voxel_count'] = 3
    payload = map_payload(scene)
    projection = payload['projection']
    assert projection['image_size'] == [1280, 720] and projection['x_limits'] == (-2., 2.)
    assert projection['viewport'] == [62, 46, 650, 650]
    assert payload['phase'] == 'skill' and payload['tracked_objects'] == 1
    assert payload['global_revision'] == 2 and payload['cloud_role'] == 'display_history_only'
    packet = encode_packet(snapshot, scene)
    json.dumps(packet, allow_nan=False)
    restored_snapshot, restored = decode_packet(packet)
    np.testing.assert_array_equal(restored['global_observed'], scene['global_observed'])
    np.testing.assert_array_equal(restored['tracked'], scene['tracked'])
    assert restored['decision'] == scene['decision'] and restored['skill'] == scene['skill']
    assert restored_snapshot.planning_frame == 'experiment'


@pytest.mark.parametrize('bounded', [False, True])
def test_detached_map_geometry_matches_direct_simulation_renderer(bounded):
    from deploy.app.config import load
    from deploy.app.demo import DemoSensors
    from deploy.app.tasks import Tasks
    from grutopia_extension.interactive_navigation.task_video_recording import TaskVideoRecorder

    tasks = Tasks({**load()['navigation'], 'world_boundary': bounded})
    snapshot = DemoSensors(SimpleNamespace(rgb=lambda *args: None)).latest()
    tasks.update(snapshot, snapshot.captured_at)
    tasks.start('navigate', (1., 0., .3), snapshot.captured_at)
    tasks.update(snapshot, snapshot.captured_at+.1)
    scene = SceneCapture().capture(snapshot, tasks)
    renderer = TaskVideoRecorder.__new__(TaskVideoRecorder)
    renderer.task = {'id': 'D01', 'start': tasks.engine.task_start, 'goal': (1., 0.)}
    renderer._trajectory = list(scene['trajectory'])
    runtime = SimpleNamespace(map=SimpleNamespace(config=tasks.engine.layers.config),
                              navigation_layers=tasks.engine.layers, _planned_path=tasks.engine.path)
    expected = renderer._render_map(0, tasks.engine.phase,
                                   SimpleNamespace(robot_position=scene['position']), runtime)
    actual = render_map(scene)
    if bounded:
        xmin, xmax = scene['world_boundary']['x_limits']
        ymin, ymax = scene['world_boundary']['y_limits']
        x0, y0 = renderer._world_to_pixel((xmin, ymax), runtime.map)
        x1, y1 = renderer._world_to_pixel((xmax, ymin), runtime.map)
        lx, ly, width, height = renderer._map_viewport
        for sample in (actual[ly:y0-3, lx:lx+width], actual[y1+4:ly+height, lx:lx+width],
                       actual[y0:y1, lx:x0-3], actual[y0:y1, x1+4:lx+width]):
            assert sample.size and np.all(sample == [164, 168, 175])
        assert actual[(y0+y1)//2, x0].tolist() == [165, 96, 32]
        np.testing.assert_array_equal(actual[y0+3:y1-2, x0+3:x1-2], expected[y0+3:y1-2, x0+3:x1-2])
    else:
        np.testing.assert_array_equal(actual[35:, :785], expected[35:, :785])
    for field, source in [('stable', 'stable_obstacles'), ('local_dynamic', 'local_dynamic'),
                          ('global_tracked', 'global_tracked')]:
        np.testing.assert_array_equal(scene[field], getattr(tasks.engine.layers, source))
        assert not np.shares_memory(scene[field], getattr(tasks.engine.layers, source))


def test_follow_cloud_height_slice_is_display_only_and_requires_calibration():
    import pytest
    from deploy.visualization.cloud_image import render_cloud
    points = np.array([[1., 0., .18], [1.5, .3, .28]])
    extra = np.vstack((points, [[1., 0., -.1], [1., 0., .2801], [1., 0., 1.2]]))
    before = extra.copy()
    arguments = dict(position=[0., 0., .3], orientation=[1., 0., 0., 0.],
                     ground_relative_z=True, mode='follow')
    expected, actual = render_cloud(points, **arguments), render_cloud(extra, **arguments)
    assert actual.shape == (720, 1280, 3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(extra, before)
    with pytest.raises(ValueError, match='ground-relative'):
        render_cloud(points, [0., 0., .3], mode='follow')
