"""Validate the optional interaction configuration without starting workers."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

from deploy.interaction.profile import ContactProfile
from deploy.robot.world_boundary import WorldBoundary


DEFAULTS = dict(source='rgbd', tracking_enabled=False, geometry_verified=False,
                boxes=[], tracking={}, contact_profile=None, max_skill_seconds=180.,
                registry_reference='', window_max_age_s=None, first_push=False)


def configure(config, payload, directory, *, registry_path=None):
    """Keep measurement/acceptance separate from choosing an observation source."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict) or set(payload)-set(DEFAULTS):
        raise ValueError('Unknown interaction configuration fields')
    result = {**deepcopy(DEFAULTS), **deepcopy(payload)}
    if result['source'] not in ('rgbd', 'registered_box_lidar'):
        raise ValueError('interaction.source must be rgbd or registered_box_lidar')
    for name in ('tracking_enabled', 'geometry_verified', 'first_push'):
        if type(result[name]) is not bool:
            raise ValueError(f'interaction.{name} must be boolean')
    window_age = result['window_max_age_s']
    if window_age is not None and (type(window_age) not in (int, float) or not .02 <= window_age <= .75):
        raise ValueError('interaction.window_max_age_s must be between .02 and .75 seconds')
    duration = result['max_skill_seconds']
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= 300:
        raise ValueError('interaction.max_skill_seconds must be in [1, 300]')
    if not isinstance(result['boxes'], list) or not isinstance(result['tracking'], dict):
        raise ValueError('interaction boxes/tracking must be a list/mapping')
    if not isinstance(result['registry_reference'], str):
        raise ValueError('interaction.registry_reference must be a string')
    if not result['registry_reference']:
        path = registry_path if registry_path is not None else Path(directory)/'interaction.yaml'
        result['registry_reference'] = str(path.resolve())+'#boxes'
    nav, robot, control = config['navigation'], config['robot'], config['runtime']['control']
    if result['first_push']:
        if (result['source'] != 'registered_box_lidar' or not result['tracking_enabled']
                or not any(row.get('push_allowed') is True for row in result['boxes'] if isinstance(row, dict))):
            raise ValueError('First push requires a registered, pushable LiDAR box')
        nav['interaction_enabled'] = True
    requested = nav['interaction_enabled']
    if result['source'] == 'registered_box_lidar':
        if result['tracking_enabled'] or requested or result['boxes']:
            from deploy.perception.registered_boxes import validate_tracking_config
            validated = validate_tracking_config(tracking_config(result))
            # Preserve only tracker knobs here; registration remains explicit.
            result['tracking'] = {k: v for k, v in validated.items()
                                  if k not in ('boxes', 'geometry_verified', 'registry_reference')}
            xlim, ylim = nav['mapping']['x_limits'], nav['mapping']['y_limits']
            for row in validated['boxes']:
                x0, x1, y0, y1 = row['search_region']
                if nav.get('world_boundary'):
                    x0, x1 = max(x0, xlim[0]), min(x1, xlim[1])
                    y0, y1 = max(y0, ylim[0]), min(y1, ylim[1])
                    row['search_region'] = [x0, x1, y0, y1]
                if not (xlim[0] <= x0 < x1 <= xlim[1] and ylim[0] <= y0 < y1 <= ylim[1]):
                    raise ValueError('Registered search_region must lie inside the navigation map')
            result['boxes'] = validated['boxes']
        if requested and not result['geometry_verified']:
            raise ValueError('Registered interaction requires verified measured box geometry and tracking')
    elif requested:
        if robot['camera_mode'] != 'rgbd' or config['runtime']['perception']['geometry_verified'] is not True:
            raise ValueError('Interaction requires RGB-D and verified complete object geometry')
    if requested and result['source'] == 'rgbd':
        raise ValueError('RGB-D provider supplies visible AABB only, not full oriented contact geometry; use registered_box_lidar for execution')
    profile = result['contact_profile']
    if isinstance(profile, str):
        path = Path(profile)
        if not path.is_absolute():
            path = Path(directory)/path
        profile = yaml.safe_load(path.read_text())
    if profile is not None:
        profile = ContactProfile.from_dict(profile)
        if result['first_push'] != (profile.mode == 'first_push'):
            raise ValueError('First-push limits can only be used in first_push mode')
        bounds = {'max_vx': max(profile.max_forward_mps, profile.max_reverse_mps),
                  'max_vy': profile.max_lateral_mps, 'max_wz': profile.max_yaw_rps,
                  'acceleration': profile.linear_acceleration_mps2,
                  'angular_acceleration': profile.yaw_acceleration_rps2}
        planner_limits = dict(max_vx=nav['max_forward_speed'], max_vy=nav['max_lateral_speed'],
                              max_wz=nav['max_yaw_speed'], acceleration=nav['linear_acceleration'],
                              angular_acceleration=nav['yaw_acceleration'])
        if any(value > min(control[key], planner_limits[key]) for key, value in bounds.items()):
            raise ValueError('Contact profile limits exceed navigation or SDK limits')
        if profile.max_tilt_degrees > control['max_tilt_degrees']:
            raise ValueError('Contact profile tilt limit exceeds the SDK guard')
        if profile.max_contact_seconds > duration:
            raise ValueError('Contact duration exceeds the whole skill budget')
        result['contact_profile'] = profile.to_dict()
        control['contact_limits'] = bounds
        control['contact_max_duration_s'] = float(duration)
    elif requested:
        raise ValueError('Interaction requires a measured SDK contact_profile in interaction.yaml')
    return result


def tracking_config(settings):
    return {**settings['tracking'], 'boxes': settings['boxes'],
            'geometry_verified': settings['geometry_verified'],
            'registry_reference': settings['registry_reference']}


def create_perception(config, *, synchronous=False, box_window=False, live=False, box_memory=None):
    """The only provider factory used by live run and deterministic replay."""
    settings = config.get('interaction', DEFAULTS)
    if box_window:
        if settings['source'] != 'registered_box_lidar' or not settings['tracking_enabled']:
            raise ValueError('Window observation requires registered_box_lidar and tracking_enabled')
        from deploy.perception.window_observer import WindowBoxObserver
        tracking = tracking_config(settings)
        if settings.get('window_max_age_s') is not None:
            tracking['max_age_s'] = settings['window_max_age_s']
        return WindowBoxObserver(tracking, live=live, world_boundary=WorldBoundary.from_navigation(config['navigation']))
    if settings['source'] == 'registered_box_lidar':
        if not (settings['tracking_enabled'] or config['navigation']['interaction_enabled']):
            return None
        from deploy.perception.registered_boxes import RegisteredBoxPerception
        return RegisteredBoxPerception(tracking_config(settings), synchronous=synchronous, box_memory=box_memory,
                                       world_boundary=WorldBoundary.from_navigation(config['navigation']))
    if (settings['tracking_enabled'] or config['navigation']['interaction_enabled']) and config['robot']['camera_mode'] == 'rgbd':
        if synchronous:
            raise ValueError('RGB-D HTTP classification is not deterministic replay; use registered LiDAR or replay observations only')
        from deploy.perception.client import AsyncInteractionPerception
        return AsyncInteractionPerception(config['runtime']['perception'])
    return None
