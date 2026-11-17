"""Display requests reuse point serialization while live metadata stays live."""
import json

import requests

from deploy.app.config import load
from deploy.app.demo import DemoSensors
from deploy.app.session import encode_display
from deploy.app.tasks import Tasks
from deploy.tests.test_session import session, wait_until


def test_cached_points_are_not_serialized_on_the_http_owner():
    cloud = {'points': [[1.25, -.5, .3]], 'frame': 'experiment', 'stamp': 10.}
    cached = json.dumps(cloud, separators=(',', ':')).encode()
    # This deliberately cannot be JSON encoded. Only the renderer's cached
    # point payload may be used, even when the live metadata changes.
    payload = {'cloud': dict(cloud, points=object(), age=1.2, stale=True),
               'status': {'state': 'paused', 'error': 'command_timeout'}, 'finished': False}
    encoded = json.loads(encode_display(payload, cached))
    assert encoded['cloud'] == dict(cloud, age=1.2, stale=True)
    assert encoded['status'] == payload['status']
    assert not encoded['finished']


def test_empty_and_legacy_display_payloads_remain_readable():
    for payload in ({'cloud': None, 'status': {'state': 'starting'}},
                    {'cloud': {'points': [[0., 0., 0.]], 'age': 0., 'stale': False}}):
        assert json.loads(encode_display(payload, None)) == payload


def test_http_reuses_rendered_geometry_but_updates_status_and_staleness(session):
    tasks = Tasks(load()['navigation'])
    snapshot = DemoSensors(session).latest()
    tasks.update(snapshot, snapshot.captured_at)
    session.update(snapshot, tasks, 'disabled')
    wait_until(lambda: session.frame_id)
    cached = json.loads(session.cloud_json)
    initial = requests.get(session.url+'/api/frame').json()
    assert initial['cloud']['points'] == cached['points']

    session.set_status('paused', 'navigate', 'stopped', error='command_timeout', map_ready=True)
    session.frames.error('cloud', 'synthetic delayed display')
    current = requests.get(session.url+'/api/frame').json()
    assert current['cloud']['points'] == initial['cloud']['points']
    assert current['cloud']['stale']
    assert current['status']['state'] == 'paused'
    assert current['status']['error'] == 'command_timeout'
    assert current['sequence'] == initial['sequence']
