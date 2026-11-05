"""Scene profiles shared by point-navigation demos and validators."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from grutopia_extension.interactive_navigation.mapping import MappingConfig, Vector3


@dataclass(frozen=True)
class PointNavigationSceneProfile:
    name: str
    start: Vector3
    goals: Tuple[Vector3, ...]
    mapping: MappingConfig
    scene_asset_path: Optional[str] = None
    scene_scale: Vector3 = (1.0, 1.0, 1.0)
    scene_position: Vector3 = (0.0, 0.0, 0.0)
    safe_base_height: float = 0.65
    fall_height: float = 0.35
    success_distance: float = 0.30
    replan_interval_steps: int = 120
    replan_lookahead_distance: float = 2.0
    use_scene_graph: bool = True
    overview_position: Vector3 = (7.0, -13.5, 8.5)
    overview_look_at: Vector3 = (7.0, 0.0, 0.7)
    overview_follow_robot: bool = False
    third_person_offset: Vector3 = (-0.70, 0.0, 1.55)
    third_person_pitch_degrees: float = 50.0
    third_person_fov_degrees: float = 80.0
    static_obstacle_metadata_path: Optional[str] = None
    static_obstacle_categories: Tuple[str, ...] = ()
    use_rgb_occupancy: bool = False
    rgb_clears_free_space: bool = False
    velocity_control: bool = False
    max_forward_speed: float = 0.75
    max_lateral_speed: float = 0.35
    apply_koostruct_material_fallbacks: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError('profile name cannot be empty')
        _vector3(self.start, 'start')
        if not self.goals:
            raise ValueError('profile must contain at least one goal')
        for index, goal in enumerate(self.goals):
            _vector3(goal, f'goals[{index}]')
        _vector3(self.scene_scale, 'scene_scale')
        _vector3(self.third_person_offset, 'third_person_offset')
        if min(self.scene_scale) <= 0:
            raise ValueError('scene_scale components must be positive')
        if self.success_distance <= 0:
            raise ValueError('success_distance must be positive')
        if self.replan_lookahead_distance <= 0:
            raise ValueError('replan_lookahead_distance must be positive')
        if self.max_forward_speed <= 0 or self.max_lateral_speed <= 0:
            raise ValueError('point-navigation speed limits must be positive')

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['mapping'] = asdict(self.mapping)
        return payload


def programmatic_g1_profile(
    environment_length: float = 16.0,
    environment_width: float = 10.0,
    start: Vector3 = (0.0, 0.0, 0.80),
    goal: Vector3 = (10.0, 2.0, 0.80),
) -> PointNavigationSceneProfile:
    if environment_length <= 2.0 or environment_width <= 2.0:
        raise ValueError('programmatic environment dimensions must exceed two metres')
    center_x = environment_length / 2.0 - 1.0
    return PointNavigationSceneProfile(
        name='programmatic_g1_household',
        start=_vector3(start, 'start'),
        goals=(_vector3(goal, 'goal'),),
        mapping=MappingConfig(
            x_limits=(-1.5, environment_length - 0.5),
            y_limits=(-environment_width / 2.0, environment_width / 2.0),
            point_height=(-0.2, 2.5),
            obstacle_height=(0.10, 1.60),
            safe_recovery_y_limits=(
                -environment_width / 2.0 + 0.5,
                environment_width / 2.0 - 0.5,
            ),
            robot_radius=0.30,
            obstacle_inflation_radius=1.00,
        ),
        overview_position=(center_x, -environment_width * 1.35, 8.5),
        overview_look_at=(center_x, 0.0, 0.7),
    )


def programmatic_go2_profile(
    environment_length: float = 16.0,
    environment_width: float = 10.0,
    start: Vector3 = (0.0, 0.0, 0.40),
    goal: Vector3 = (3.0, 0.0, 0.40),
) -> PointNavigationSceneProfile:
    if environment_length <= 2.0 or environment_width <= 2.0:
        raise ValueError(
            'programmatic environment dimensions must exceed two metres'
        )
    center_x = environment_length / 2.0 - 1.0
    return PointNavigationSceneProfile(
        name='programmatic_go2_household',
        start=_vector3(start, 'start'),
        goals=(_vector3(goal, 'goal'),),
        mapping=MappingConfig(
            x_limits=(-1.5, environment_length - 0.5),
            y_limits=(-environment_width / 2.0, environment_width / 2.0),
            point_height=(-0.1, 2.0),
            obstacle_height=(0.05, 1.20),
            safe_recovery_y_limits=(
                -environment_width / 2.0 + 0.5,
                environment_width / 2.0 - 0.5,
            ),
            robot_radius=0.30,
            obstacle_inflation_radius=0.75,
        ),
        safe_base_height=0.22,
        fall_height=0.12,
        success_distance=0.70,
        replan_interval_steps=200,
        overview_position=(center_x, -environment_width * 1.35, 7.0),
        overview_look_at=(center_x, 0.0, 0.35),
        overview_follow_robot=True,
        third_person_offset=(-1.8, 0.0, 0.7),
        third_person_pitch_degrees=18.0,
        third_person_fov_degrees=70.0,
        velocity_control=False,
        max_forward_speed=0.80,
        max_lateral_speed=0.20,
    )


def load_point_navigation_profile(path: str) -> PointNavigationSceneProfile:
    profile_path = Path(path)
    with profile_path.open('r', encoding='utf-8') as profile_file:
        data = json.load(profile_file)
    if not isinstance(data, Mapping):
        raise ValueError('point-navigation profile root must be a JSON object')
    mapping_data = data.get('mapping')
    if not isinstance(mapping_data, Mapping):
        raise ValueError('profile mapping must be a JSON object')
    mapping_values = dict(mapping_data)
    for key in (
        'x_limits',
        'y_limits',
        'obstacle_height',
        'point_height',
        'safe_recovery_y_limits',
    ):
        if key in mapping_values:
            mapping_values[key] = tuple(float(value) for value in mapping_values[key])
    scene_asset_path = data.get('scene_asset_path')
    if scene_asset_path:
        scene_asset_path = str(Path(scene_asset_path).expanduser())
    return PointNavigationSceneProfile(
        name=str(data['name']),
        start=_vector3(data['start'], 'start'),
        goals=tuple(_vector3(goal, 'goal') for goal in data['goals']),
        mapping=MappingConfig(**mapping_values),
        scene_asset_path=scene_asset_path,
        scene_scale=_vector3(data.get('scene_scale', (1.0, 1.0, 1.0)), 'scene_scale'),
        scene_position=_vector3(data.get('scene_position', (0.0, 0.0, 0.0)), 'scene_position'),
        safe_base_height=float(data.get('safe_base_height', 0.65)),
        fall_height=float(data.get('fall_height', 0.35)),
        success_distance=float(data.get('success_distance', 0.30)),
        replan_interval_steps=int(data.get('replan_interval_steps', 120)),
        replan_lookahead_distance=float(data.get('replan_lookahead_distance', 2.0)),
        use_scene_graph=bool(data.get('use_scene_graph', True)),
        overview_position=_vector3(data.get('overview_position', (7.0, -13.5, 8.5)), 'overview_position'),
        overview_look_at=_vector3(data.get('overview_look_at', (7.0, 0.0, 0.7)), 'overview_look_at'),
        overview_follow_robot=bool(data.get('overview_follow_robot', False)),
        third_person_offset=_vector3(
            data.get('third_person_offset', (-0.70, 0.0, 1.55)),
            'third_person_offset',
        ),
        third_person_pitch_degrees=float(data.get('third_person_pitch_degrees', 50.0)),
        third_person_fov_degrees=float(data.get('third_person_fov_degrees', 80.0)),
        static_obstacle_metadata_path=data.get('static_obstacle_metadata_path'),
        static_obstacle_categories=tuple(str(value) for value in data.get('static_obstacle_categories', ())),
        use_rgb_occupancy=bool(data.get('use_rgb_occupancy', False)),
        rgb_clears_free_space=bool(data.get('rgb_clears_free_space', False)),
        velocity_control=bool(data.get('velocity_control', False)),
        max_forward_speed=float(data.get('max_forward_speed', 0.75)),
        max_lateral_speed=float(data.get('max_lateral_speed', 0.35)),
        apply_koostruct_material_fallbacks=bool(data.get('apply_koostruct_material_fallbacks', False)),
    )


def load_profile_static_obstacles(profile: PointNavigationSceneProfile):
    if not profile.static_obstacle_metadata_path or not profile.static_obstacle_categories:
        return ()
    metadata_path = Path(profile.static_obstacle_metadata_path)
    with metadata_path.open('r', encoding='utf-8') as metadata_file:
        objects = json.load(metadata_file)
    categories = set(profile.static_obstacle_categories)
    obstacles = []
    for instance_id, metadata in objects.items():
        if metadata.get('category') not in categories:
            continue
        minimum = metadata.get('min_points')
        maximum = metadata.get('max_points')
        if not isinstance(minimum, list) or not isinstance(maximum, list):
            continue
        if maximum[2] < profile.mapping.obstacle_height[0] or minimum[2] > profile.mapping.obstacle_height[1]:
            continue
        obstacles.append(
            {
                'label': instance_id,
                'minimum_xy': tuple(float(value) for value in minimum[:2]),
                'maximum_xy': tuple(float(value) for value in maximum[:2]),
            }
        )
    return tuple(obstacles)


def save_point_navigation_profile(profile: PointNavigationSceneProfile, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(profile.to_dict(), output_file, indent=2)


def _vector3(value, name: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must contain three numbers')
    return tuple(float(component) for component in value)
