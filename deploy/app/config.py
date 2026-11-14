"""Load deployment settings and the optional real-interaction profile."""
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load(directory=None):
    directory = Path(directory) if directory else ROOT / 'configs'
    result = {}
    for name in ('robot', 'navigation', 'runtime'):
        with (directory / f'{name}.yaml').open() as stream:
            result[name] = yaml.safe_load(stream)
        if not isinstance(result[name], dict):
            raise ValueError(f'{name}.yaml must contain a mapping')
    def validate(value, path='config'):
        if isinstance(value, dict):
            for key, item in value.items():
                validate(item, f'{path}.{key}')
        elif isinstance(value, list):
            for item in value:
                validate(item, path)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{path}: non-finite number')
    validate(result)
    from dataclasses import fields
    from grutopia_extension.interactive_navigation.mapping import MappingConfig
    unknown = set(result['navigation']['mapping']) - {f.name for f in fields(MappingConfig)}
    retired = set(result['navigation']) & {'direct_path_tracking_margin', 'visual_observation_budget_s',
        'frontier_selection_s', 'frontier_progress_timeout_s', 'frontier_goal_timeout_s',
        'frontier_scan_timeout_s', 'fresh_target_observations', 'topology_interval_s'}
    if unknown or retired or 'topology_scorer' in result['runtime']:
        raise ValueError('Retired deploy configuration; migrate from deploy/configs: ' +
                         ', '.join(sorted(unknown | retired | ({'topology_scorer'} if 'topology_scorer' in result['runtime'] else set()))))
    positive = {
        'robot': ('max_data_age_s', 'image_sync_tolerance_s', 'lidar_max_range',
                  'max_position_jump_m', 'max_orientation_jump_rad', 'pose_cloud_tolerance_s'),
        'navigation': ('tick_hz', 'task_timeout_s', 'max_forward_speed', 'max_lateral_speed',
                       'success_distance', 'final_goal_slow_radius', 'safe_base_height', 'fall_height',
                       'max_yaw_speed', 'route_clearance', 'navigation_half_width', 'local_map_hz', 'global_map_hz',
                       'local_map_radius', 'object_max_age_s', 'recognition_max_age_s', 'push_start_speed',
                       'push_max_speed', 'subnavigation_timeout_s', 'inspection_timeout_s',
                       'linear_acceleration', 'yaw_acceleration'),
    }
    for group, keys in positive.items():
        for key in keys:
            if key not in result[group]:
                raise ValueError(f'Missing {group}.{key}; migrate from deploy/configs')
            if type(result[group][key]) not in (int, float) or float(result[group][key]) <= 0:
                raise ValueError(f'{group}.{key} must be positive')
    for key, value in result['runtime']['control'].items():
        if key in ('contact_limits', 'contact_max_duration_s'):
            continue  # Rebuilt from the validated interaction profile below.
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'control.{key} must be positive')
    control = result['runtime']['control']
    if not 0 < control['max_tilt_degrees'] < 90 or control['stop_settle_s'] >= control['stop_timeout_s']:
        raise ValueError('Invalid tilt limit or stopping time configuration')
    for key in ('request_timeout', 'qwen_vl_timeout', 'result_max_age_s', 'geometry_max_age_s', 'input_max_age_s'):
        if key not in result['runtime']['perception']:
            raise ValueError(f'Missing perception.{key}; migrate from deploy/configs')
        value = result['runtime']['perception'][key]
        if type(value) not in (int, float) or value <= 0:
            raise ValueError(f'perception.{key} must be positive')
    if type(result['navigation']['interaction_enabled']) is not bool:
        raise ValueError('navigation.interaction_enabled must be boolean')
    if type(result['runtime']['perception']['geometry_verified']) is not bool:
        raise ValueError('perception.geometry_verified must be boolean')
    if result['robot']['map_frame'] == result['robot']['base_frame']:
        raise ValueError('map_frame and base_frame must differ')
    robot = result['robot']
    if robot['camera_mode'] not in ('unavailable', 'rgbd'):
        raise ValueError('camera_mode must be unavailable or rgbd')
    if robot['camera_mode'] == 'rgbd' and not all(
            robot[key].strip() for key in ('rgb_topic', 'depth_topic', 'camera_info_topic')):
        raise ValueError('RGB-D requires explicit RGB, aligned depth and CameraInfo topics')
    offset = robot['lidar_origin_offset_base_m']
    if offset is not None and (not isinstance(offset, list) or len(offset) != 3
                              or not all(type(v) in (int, float) and math.isfinite(v) for v in offset)):
        raise ValueError('lidar_origin_offset_base_m must be null or three finite metres')
    if robot['ground_z'] is not None and type(robot['ground_z']) not in (int, float):
        raise ValueError('ground_z must be null or a finite measured height')
    if 'sport_state_motion_codes' in robot:
        codes = robot['sport_state_motion_codes']
        if (not isinstance(codes, list) or not codes
                or not all(type(code) is int and 0 <= code <= 0xffffffff for code in codes)):
            raise ValueError('sport_state_motion_codes must be a nonempty list of reviewed uint32 state values')
    nav, control = result['navigation'], result['runtime']['control']
    if nav['max_forward_speed'] > control['max_vx'] or nav['max_lateral_speed'] > control['max_vy'] or nav['max_yaw_speed'] > control['max_wz']:
        raise ValueError('Navigation velocity limits exceed SDK control limits')
    if nav['linear_acceleration'] > control['acceleration'] or nav['yaw_acceleration'] > control['angular_acceleration']:
        raise ValueError('Navigation acceleration limits exceed SDK control limits')
    if nav['global_map_hz'] > nav['local_map_hz'] or nav['local_map_hz'] > nav['tick_hz']:
        raise ValueError('Map rates must satisfy global_map_hz <= local_map_hz <= tick_hz')
    if nav['success_distance'] > .25 or nav['push_start_speed'] > nav['push_max_speed']:
        raise ValueError('Invalid goal tolerance or push speed range')
    from deploy.interaction.config import configure
    path = directory/'interaction.yaml'
    payload = yaml.safe_load(path.read_text()) if path.exists() else None
    control.pop('contact_limits', None)
    control.pop('contact_max_duration_s', None)
    result['interaction'] = configure(result, payload, directory)
    return result


def check_control_configuration(config, *, require_takeover=True):
    """SDK prerequisites, independent of map and full-navigation acceptance."""
    robot = config['robot']
    if not robot['network_interface'].strip():
        raise ValueError('Fill robot.yaml network_interface before enabling motion')
    mask = robot['takeover_button_mask']
    if type(mask) is not int or not 0 <= mask <= 65535:
        raise ValueError('takeover_button_mask must be an integer from 0 to 65535')
    if require_takeover and mask == 0:
        raise ValueError('Configure a verified takeover_button_mask; --motion-test can display button bits')
    if not robot.get('sport_state_motion_codes'):
        raise ValueError('Configure sport_state_motion_codes for this robot before enabling motion')


def check_motion_configuration(config):
    """Full autonomous navigation requires both SDK prerequisites and field acceptance."""
    robot = config['robot']
    for field in ('calibration_verified', 'deskew_and_self_filter_verified', 'communication_verified', 'remote_arbitration_verified'):
        if robot[field] is not True:
            raise ValueError(f'Motion disabled: robot.yaml {field} is not verified; use --motion-test for bounded acceptance trials')
    if robot['ground_z'] is None or robot['lidar_origin_offset_base_m'] is None:
        raise ValueError('Measure ground_z and lidar_origin_offset_base_m before enabling navigation')
    check_control_configuration(config)


def check_task_inputs(robot, snapshot):
    if not snapshot.geometry_calibrated:
        raise ValueError('尚未测量 ground_z / lidar_origin_offset_base_m：当前仅记录原始观测，不建导航地图')
