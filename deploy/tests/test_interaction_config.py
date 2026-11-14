"""Source selection, measurement requirements and shared live/replay factory."""
from copy import deepcopy
import yaml
import pytest

from deploy.app.config import load
from deploy.interaction.config import create_perception
from deploy.tests.test_interaction_execution import profile
from deploy.tests.test_registered_boxes import config as tracking_config


def configured(tmp_path, *, active=False):
    data = load()
    tracking = tracking_config()
    data['navigation']['interaction_enabled'] = active
    data['robot']['interaction_motion_verified'] = active
    data['interaction'].update(source='registered_box_lidar', tracking_enabled=True,
        geometry_verified=True, boxes=tracking['boxes'], contact_profile=profile().to_dict())
    for name, value in data.items():
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(value))
    return data


def write(tmp_path, data):
    for name, value in data.items():
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(value))
    return load(tmp_path)


def test_registered_interaction_does_not_require_rgbd_or_models(tmp_path):
    configured(tmp_path, active=True)
    loaded = load(tmp_path)
    assert loaded['robot']['camera_mode'] == 'unavailable'
    assert loaded['runtime']['control']['contact_limits']['max_vx'] == .25
    provider = create_perception(loaded, synchronous=True)
    try:
        assert provider.source == 'registered_box_lidar' and provider.thread is None
    finally:
        provider.close()


def test_tracking_can_run_unverified_but_cannot_enable_motion(tmp_path):
    data = configured(tmp_path)
    data['interaction']['geometry_verified'] = False
    loaded = write(tmp_path, data)
    provider = create_perception(loaded, synchronous=True)
    try:
        assert not provider.geometry_verified
    finally:
        provider.close()
    data['navigation']['interaction_enabled'] = True
    with pytest.raises(ValueError, match='verified measured box geometry'):
        write(tmp_path, data)


@pytest.mark.parametrize('setting', ['contact_profile', 'boxes'])
def test_missing_measurements_cannot_be_replaced_by_an_enable_flag(tmp_path, setting):
    data = configured(tmp_path, active=True)
    data['interaction'][setting] = None if setting == 'contact_profile' else []
    with pytest.raises(ValueError):
        write(tmp_path, data)


def test_profile_path_is_resolved_relative_to_config_directory(tmp_path):
    data = configured(tmp_path, active=True)
    (tmp_path/'measured.yaml').write_text(yaml.safe_dump(data['interaction']['contact_profile']))
    data['interaction']['contact_profile'] = 'measured.yaml'
    loaded = write(tmp_path, data)
    assert loaded['interaction']['contact_profile']['backend'] == 'sdk'


def test_partial_profile_cannot_inherit_simulation_calibration(tmp_path):
    data = configured(tmp_path, active=True)
    del data['interaction']['contact_profile']['nose_offset_m']
    with pytest.raises(ValueError, match='explicit fields'):
        write(tmp_path, data)


def test_contact_limits_cannot_exceed_navigation_or_writer(tmp_path):
    data = configured(tmp_path, active=True)
    data['interaction']['contact_profile']['max_forward_mps'] = 1.0
    with pytest.raises(ValueError, match='limits exceed'):
        write(tmp_path, data)


def test_no_tracking_worker_with_default_navigation_configuration():
    assert create_perception(load()) is None


def test_missing_optional_interaction_file_keeps_plain_navigation(tmp_path):
    data = load()
    for name in ('robot', 'navigation', 'runtime'):
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(data[name]))
    loaded = load(tmp_path)
    assert loaded['interaction']['source'] == 'rgbd'
    assert create_perception(loaded) is None


def test_no_opaque_control_caps_can_override_the_measured_profile(tmp_path):
    data = configured(tmp_path, active=True)
    data['runtime']['control']['contact_limits'] = {'max_vx': 1000.}
    loaded = write(tmp_path, data)
    assert loaded['runtime']['control']['contact_limits']['max_vx'] == .25


def test_actual_cli_replays_registered_clouds_and_records_pose_without_sdk(tmp_path):
    import json
    import subprocess
    import sys
    from deploy.app.config import REPO
    from deploy.recording.recorder import Recorder
    from deploy.tests.test_registered_interaction_integration import scan, settings
    directory = tmp_path/'configs'
    directory.mkdir()
    data = configured(directory)
    _, _, tracking = settings()
    data['interaction']['boxes'] = tracking['boxes']
    loaded = write(directory, data)
    recorder = Recorder(tmp_path/'input', loaded, 'synthetic_registered_clouds_not_physics', REPO)
    for index in range(1, 5):
        recorder.observation(scan(index, 1000.+index*.1))
    recorder.close({'hardware_validated': False})
    output = tmp_path/'output'
    process = subprocess.run([sys.executable, str(REPO/'deploy/run.py'), '--config-dir', str(directory),
        '--replay', str(recorder.directory), '--output', str(output)], cwd='/tmp',
        capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout+process.stderr
    summary = next(output.glob('*/run_summary.json'))
    assert json.loads(summary.read_text())['sent_to_robot'] is False
    records = [json.loads(line) for line in (summary.parent/'events.jsonl').read_text().splitlines()]
    deliveries = [event for event in records if event['event'] == 'perception_result']
    assert any(row['valid'] and row['object_id'] == 'B1' for event in deliveries for row in event['detections'])
