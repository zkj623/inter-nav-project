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


@pytest.mark.parametrize('enabled', [True, False])
def test_only_registered_lidar_can_supply_box_evidence(tmp_path, enabled):
    config = load()
    config['interaction']['source'] = 'rgbd'
    config['navigation']['interaction_enabled'] = enabled
    with pytest.raises(ValueError, match='registered_box_lidar'):
        load(write_config(tmp_path, config))


@pytest.mark.parametrize('profile', ['mid360s_lio_utlidar', 'mid360s_raw_lio', 'l1', None])
def test_only_registered_mid360_input_is_supported(tmp_path, profile):
    config = load()
    config['robot']['sensor_profile'] = profile
    with pytest.raises(ValueError, match='mid360s_lio'):
        load(write_config(tmp_path, config))
