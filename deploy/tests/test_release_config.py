"""Configuration migration must not silently reuse the retired planner knobs."""
import yaml
import pytest
from deploy.app.config import load


def write_config(tmp_path, config):
    for name, value in config.items():
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(value))
    return tmp_path


def test_old_deploy_configuration_reports_migration_not_key_error(tmp_path):
    config = load()
    config['navigation']['direct_path_tracking_margin'] = .10
    config['navigation']['mapping']['lidar_height_resolution'] = .10
    del config['navigation']['local_map_hz']
    write_config(tmp_path, config)
    with pytest.raises(ValueError, match='Retired deploy configuration; migrate'):
        load(tmp_path)


def test_missing_new_field_reports_config_migration(tmp_path):
    config = load()
    del config['navigation']['global_map_hz']
    with pytest.raises(ValueError, match='Missing navigation.global_map_hz; migrate'):
        load(write_config(tmp_path, config))


@pytest.mark.parametrize('field,value', [('max_yaw_speed', 1.1), ('linear_acceleration', .7)])
def test_planner_limits_cannot_exceed_real_command_guard(tmp_path, field, value):
    config = load()
    config['navigation'][field] = value
    with pytest.raises(ValueError, match='limits exceed SDK'):
        load(write_config(tmp_path, config))


def test_single_camera_preview_cannot_enable_interaction(tmp_path):
    config = load()
    config['interaction']['source'] = 'rgbd'
    config['navigation']['interaction_enabled'] = True
    with pytest.raises(ValueError, match='RGB-D'):
        load(write_config(tmp_path, config))


def test_rgbd_alone_does_not_bypass_contact_motion_acceptance(tmp_path):
    config = load()
    config['interaction']['source'] = 'rgbd'
    config['navigation']['interaction_enabled'] = True
    config['robot'].update(camera_mode='rgbd', rgb_topic='/rgb', depth_topic='/depth', camera_info_topic='/info')
    config['runtime']['perception']['geometry_verified'] = True
    with pytest.raises(ValueError, match='SDK contact-motion acceptance'):
        load(write_config(tmp_path, config))
    config['robot']['interaction_motion_verified'] = True
    with pytest.raises(ValueError, match='full oriented contact geometry'):
        load(write_config(tmp_path, config))
