"""Load the three explicit deployment configurations."""
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
    positive = {
        'robot': ('max_data_age_s', 'image_sync_tolerance_s', 'lidar_max_range',
                  'max_position_jump_m', 'max_orientation_jump_rad', 'pose_cloud_tolerance_s'),
        'navigation': ('tick_hz', 'task_timeout_s', 'max_forward_speed', 'max_lateral_speed',
                       'success_distance', 'safe_base_height', 'fall_height'),
    }
    for group, keys in positive.items():
        for key in keys:
            if float(result[group][key]) <= 0:
                raise ValueError(f'{group}.{key} must be positive')
    for key, value in result['runtime']['control'].items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'control.{key} must be positive')
    control = result['runtime']['control']
    if not 0 < control['max_tilt_degrees'] < 90 or control['stop_settle_s'] >= control['stop_timeout_s']:
        raise ValueError('Invalid tilt limit or stopping time configuration')
    if result['navigation']['fresh_target_observations'] < 1:
        raise ValueError('fresh_target_observations must be positive')
    if result['robot']['map_frame'] == result['robot']['base_frame']:
        raise ValueError('map_frame and base_frame must differ')
    for key in ('topology_interval_s', 'frontier_selection_s', 'frontier_progress_timeout_s',
                'frontier_goal_timeout_s', 'frontier_scan_timeout_s', 'visual_observation_budget_s'):
        if result['navigation'][key] <= 0:
            raise ValueError(f'navigation.{key} must be positive')
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
    return result


def check_motion_configuration(config):
    robot = config['robot']
    if not robot['network_interface'].strip():
        raise ValueError('Fill robot.yaml network_interface before enabling motion')
    for field in ('calibration_verified', 'deskew_and_self_filter_verified', 'communication_verified', 'remote_arbitration_verified'):
        if robot[field] is not True:
            raise ValueError(f'Motion disabled: robot.yaml {field} is not verified')
    if robot['ground_z'] is None or robot['lidar_origin_offset_base_m'] is None:
        raise ValueError('Measure ground_z and lidar_origin_offset_base_m before enabling motion')
    if not 0 < int(robot['takeover_button_mask']) <= 65535:
        raise ValueError('Configure a verified takeover_button_mask')
    if not robot.get('sport_state_motion_codes'):
        raise ValueError('Configure sport_state_motion_codes for this robot before enabling motion')


def check_task_inputs(robot, snapshot, exploring=False):
    if not snapshot.geometry_calibrated:
        raise ValueError('尚未测量 ground_z / lidar_origin_offset_base_m：当前仅记录原始观测，不建导航地图')
    if exploring and robot['camera_mode'] != 'rgbd':
        raise ValueError('语义找物未就绪：前置视频不是 RGB-D；需要对齐深度、内参和相机外参')
