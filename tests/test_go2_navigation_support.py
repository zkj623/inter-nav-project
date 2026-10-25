import importlib.util
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from grutopia_extension.configs.robots.go2 import GO2_JOINT_NAMES
from grutopia_extension.interactive_navigation.mapping_runtime import (
    _NON_OBJECT_SEMANTIC_LABELS,
)
from grutopia_extension.interactive_navigation.output_paths import resolve_map_output
from grutopia_extension.interactive_navigation.point_navigation_profiles import (
    programmatic_go2_profile,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        _ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ASSET_MODULE = _load_module(
    'go2_asset_for_test',
    'grutopia_extension/robots/go2_asset.py',
)
_POLICY_MODULE = _load_module(
    'go2_policy_for_test',
    'grutopia_extension/controllers/go2_policy.py',
)
GO2_DEFAULT_JOINT_POSITIONS = _ASSET_MODULE.GO2_DEFAULT_JOINT_POSITIONS
build_fallback_go2_urdf = _ASSET_MODULE.build_fallback_go2_urdf
official_go2_meshes_ready = _ASSET_MODULE.official_go2_meshes_ready
prepare_official_go2_urdf = _ASSET_MODULE.prepare_official_go2_urdf
is_isaaclab_go2_usd = _ASSET_MODULE.is_isaaclab_go2_usd
FLAT_HIDDEN_DIMS = _POLICY_MODULE.FLAT_HIDDEN_DIMS
FLAT_OBSERVATION_DIM = _POLICY_MODULE.FLAT_OBSERVATION_DIM
HEIGHT_SCAN_DIM = _POLICY_MODULE.HEIGHT_SCAN_DIM
ROUGH_HIDDEN_DIMS = _POLICY_MODULE.ROUGH_HIDDEN_DIMS
ROUGH_OBSERVATION_DIM = _POLICY_MODULE.ROUGH_OBSERVATION_DIM
build_go2_actor = _POLICY_MODULE.build_go2_actor
build_go2_observation = _POLICY_MODULE.build_go2_observation
extract_actor_state = _POLICY_MODULE.extract_actor_state
flat_ground_height_scan = _POLICY_MODULE.flat_ground_height_scan
infer_actor_hidden_dims = _POLICY_MODULE.infer_actor_hidden_dims
infer_observation_dim = _POLICY_MODULE.infer_observation_dim
load_go2_actor = _POLICY_MODULE.load_go2_actor


class Go2NavigationSupportTest(unittest.TestCase):
    def test_semantic_map_output_defaults_to_recording_directory(self):
        self.assertEqual(
            resolve_map_output('grutopia/results/open_vocab', None),
            'grutopia/results/open_vocab/final_map',
        )
        self.assertEqual(resolve_map_output('', None), '')
        self.assertEqual(
            resolve_map_output('ignored', 'custom/final_map'),
            'custom/final_map',
        )

    def test_fallback_asset_has_expected_policy_joint_contract(self):
        root = ET.fromstring(build_fallback_go2_urdf())
        joint_names = [
            joint.attrib['name']
            for joint in root.findall('joint')
            if joint.attrib['type'] == 'revolute'
        ]

        self.assertCountEqual(joint_names, GO2_JOINT_NAMES)
        self.assertEqual(len(GO2_DEFAULT_JOINT_POSITIONS), 12)
        self.assertIsNotNone(root.find("./link[@name='base']"))
        self.assertIsNotNone(root.find("./link[@name='Head_upper']"))
        self.assertEqual(root.findall('.//mesh'), [])

    def test_go2_profile_uses_quadruped_height_and_waypoint_control(self):
        profile = programmatic_go2_profile()

        self.assertEqual(profile.start[2], 0.40)
        self.assertLess(profile.fall_height, profile.safe_base_height)
        self.assertFalse(profile.velocity_control)
        self.assertLess(profile.mapping.obstacle_height[0], profile.start[2])
        self.assertEqual(profile.name, 'programmatic_go2_household')
        self.assertGreaterEqual(profile.max_forward_speed, 0.80)

    def test_go2_semantics_are_not_mapped_as_obstacles(self):
        self.assertIn('go2', _NON_OBJECT_SEMANTIC_LABELS)
        self.assertIn('unitree_go2', _NON_OBJECT_SEMANTIC_LABELS)

    def test_official_urdf_rewrites_package_mesh_uris(self):
        asset_dir = (
            Path(__file__).resolve().parents[1]
            / 'grutopia'
            / 'assets'
            / 'robots'
            / 'go2'
        )
        if not (asset_dir / 'go2_description.urdf').is_file():
            self.skipTest('official Go2 URDF is not installed')
        rewritten = prepare_official_go2_urdf(asset_dir).read_text(encoding='utf-8')
        self.assertNotIn('package://go2_description/dae/', rewritten)
        self.assertIn('/dae/', rewritten)
        if official_go2_meshes_ready(asset_dir):
            self.assertIn('base.dae', rewritten)

    def test_rough_policy_observation_matches_isaac_lab_go2(self):
        scan = flat_ground_height_scan(0.40)
        self.assertEqual(scan.shape, (HEIGHT_SCAN_DIM,))
        self.assertTrue(((scan > -0.11) & (scan < -0.09)).all())

        observation = build_go2_observation(
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -1.0],
            [0.8, 0.0, 0.0],
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            [0.0] * 12,
            [0.0] * 12,
            ROUGH_OBSERVATION_DIM,
            base_height=0.40,
        )
        self.assertEqual(observation.shape, (ROUGH_OBSERVATION_DIM,))
        self.assertEqual(observation[48:].shape, (HEIGHT_SCAN_DIM,))

        actor = build_go2_actor(ROUGH_OBSERVATION_DIM, ROUGH_HIDDEN_DIMS)
        actor_state = actor.state_dict()
        self.assertEqual(infer_observation_dim(actor_state), ROUGH_OBSERVATION_DIM)
        self.assertEqual(infer_actor_hidden_dims(actor_state), ROUGH_HIDDEN_DIMS)
        extracted = extract_actor_state({'model_state_dict': {
            f'actor.{key}': value for key, value in actor_state.items()
        }})
        self.assertEqual(set(extracted), set(actor_state))
        self.assertEqual(
            infer_actor_hidden_dims(
                build_go2_actor(FLAT_OBSERVATION_DIM, FLAT_HIDDEN_DIMS).state_dict()
            ),
            FLAT_HIDDEN_DIMS,
        )

    def test_height_scan_subtracts_elevated_ground(self):
        # Standing 0.40 above a floor whose top surface is at world z=0.15
        # must produce the same scan as standing on a floor at z=0.
        elevated = flat_ground_height_scan(0.55, ground_height=0.15)
        flat = flat_ground_height_scan(0.40)
        self.assertTrue((elevated == flat).all())
        low_floor = flat_ground_height_scan(0.41, ground_height=0.01)
        self.assertTrue((low_floor == flat).all())
        # A stale ground height of 0 reads the terrain 15 cm too low.
        stale = flat_ground_height_scan(0.55)
        self.assertTrue(((stale - elevated) > 0.14).all())

    def test_controller_config_carries_ground_height(self):
        from grutopia_extension.configs.robots.go2 import (
            go2_navigation_controller_cfgs,
        )

        move_by_speed, move_along_path = go2_navigation_controller_cfgs(
            'unused.pt',
            ground_height=0.15,
        )
        self.assertEqual(move_by_speed.ground_height, 0.15)
        nested = move_along_path.sub_controllers[0].sub_controllers[0]
        self.assertEqual(nested.ground_height, 0.15)

    def test_semantic_exploration_repackaging_keeps_ground_height(self):
        # run_go2_semantic_exploration rebuilds a Go2NavigationRunConfig from
        # its own run config; dropping ground_height there silently reverts
        # the height scan to a z=0 floor.
        source = (
            _ROOT
            / 'grutopia_extension'
            / 'interactive_navigation'
            / 'go2_navigation_runner.py'
        ).read_text(encoding='utf-8')
        self.assertIn('ground_height=run.ground_height', source)
        self.assertIn(
            'ground_height=getattr(run, \'ground_height\', 0.0)',
            source,
        )

    def test_installed_rough_checkpoint_matches_isaac_go2_ros2(self):
        checkpoint = (
            _ROOT
            / 'grutopia'
            / 'assets'
            / 'robots'
            / 'go2'
            / 'policy'
            / 'move_by_speed'
            / 'rough_model_7850.pt'
        )
        if not checkpoint.is_file():
            self.skipTest('rough Go2 checkpoint is not installed')
        actor = load_go2_actor(str(checkpoint))
        self.assertEqual(actor[0].in_features, ROUGH_OBSERVATION_DIM)
        self.assertEqual(actor[-1].out_features, 12)
        self.assertEqual(
            [layer.out_features for layer in actor if hasattr(layer, 'out_features')][:-1],
            list(ROUGH_HIDDEN_DIMS),
        )

    def test_isaaclab_go2_usd_is_nvidia_packed_usdc(self):
        usd_path = (
            _ROOT
            / 'grutopia'
            / 'assets'
            / 'robots'
            / 'go2'
            / 'isaaclab_go2.usd'
        )
        if not usd_path.is_file():
            self.skipTest('Isaac Lab Go2 USD is not installed')
        self.assertTrue(is_isaaclab_go2_usd(usd_path))
        self.assertGreater(usd_path.stat().st_size, 20_000_000)

    def test_constant_walk_demo_uses_empty_floor(self):
        source = (
            _ROOT / 'grutopia' / 'demo' / 'go2_walk_demo.py'
        ).read_text(encoding='utf-8')
        self.assertIn('FixedCubeCfg', source)
        self.assertNotIn('ProceduralHouseholdCfg', source)

    def test_semantic_exploration_uses_far_household_landmarks(self):
        import math

        from grutopia_extension.configs.objects import (
            HouseholdSemanticPropsCfg,
            ProceduralHouseholdCfg,
        )
        from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
            OpenVocabularyPerceptionConfig,
        )

        props = HouseholdSemanticPropsCfg(
            name='household_semantic_props_0',
            prim_path='/World/env_0/objects/household_semantic_props',
        )
        fridge_range = math.hypot(*props.refrigerator_position[:2])
        source = (
            _ROOT / 'grutopia' / 'demo' / 'go2_semantic_exploration.py'
        ).read_text(encoding='utf-8')
        household_source = (
            _ROOT / 'grutopia_extension' / 'objects' / 'procedural_household.py'
        ).read_text(encoding='utf-8')

        household = ProceduralHouseholdCfg(
            name='interactive_household_0',
            prim_path='/World/env_0/objects/interactive_household',
            include_carry_object=False,
        )
        self.assertFalse(household.include_carry_object)
        self.assertTrue(
            ProceduralHouseholdCfg(
                name='interactive_household_0',
                prim_path='/World/env_0/objects/interactive_household',
            ).include_carry_object
        )
        self.assertGreater(fridge_range, 11.5)
        self.assertGreater(math.hypot(*props.chair_position[:2]), 6.0)
        self.assertGreater(math.hypot(*props.plant_position[:2]), 9.0)
        self.assertIn('plant', OpenVocabularyPerceptionConfig().context_categories)
        self.assertIn('refrigerator', OpenVocabularyPerceptionConfig().context_categories)
        self.assertIn('HouseholdSemanticPropsCfg', source)
        self.assertIn("default='refrigerator'", source)
        self.assertIn("choices=('grscene', 'programmatic')", source)
        self.assertIn('build_grscene_floor', source)
        self.assertIn("update={'include_carry_object': False}", source)
        self.assertNotIn("default='carry object'", source)
        self.assertIn('if config.include_carry_object:', household_source)

    def test_semantic_exploration_grscene_start_is_far_from_refrigerator(self):
        import math

        from grutopia_extension.interactive_navigation.point_navigation_profiles import (
            load_point_navigation_profile,
            load_profile_static_obstacles,
        )

        profile = load_point_navigation_profile(
            _ROOT / 'grutopia' / 'demo' / 'profiles' / 'go2_grscene_mv7_exploration.json'
        )
        fridge = (10.959860905091203, -1.7563874653050138)
        if not Path(profile.scene_asset_path).is_file() or not Path(profile.static_obstacle_metadata_path).is_file():
            self.skipTest('GRScenes asset and collision metadata are required for this integration check')
        obstacles = load_profile_static_obstacles(profile)

        self.assertTrue(Path(profile.scene_asset_path).is_file())
        self.assertGreater(math.hypot(profile.start[0] - fridge[0], profile.start[1] - fridge[1]), 8.5)
        self.assertGreaterEqual(profile.start[2], 0.50)
        # The obstacle band must clear the flat collision floor (top at
        # 0.10 m) yet stay below low furniture seats (about 0.34 m).
        self.assertGreaterEqual(profile.mapping.obstacle_height[0], 0.14)
        self.assertLessEqual(profile.mapping.obstacle_height[0], 0.30)
        self.assertIn('refrigerator', profile.static_obstacle_categories)
        self.assertGreaterEqual(len(obstacles), 10)
        self.assertLess(profile.success_distance, 1.5)

    def test_go2_first_person_camera_clears_the_snout(self):
        from grutopia_extension.interactive_navigation.navigation_sensors import (
            GO2_FIRST_PERSON_OFFSET,
            GO2_FIRST_PERSON_PITCH_DEGREES,
        )

        self.assertGreater(GO2_FIRST_PERSON_OFFSET[0], 0.35)
        self.assertGreater(GO2_FIRST_PERSON_OFFSET[2], 0.10)
        self.assertLess(GO2_FIRST_PERSON_PITCH_DEGREES, 8.0)
        runner_source = (
            _ROOT
            / 'grutopia_extension'
            / 'interactive_navigation'
            / 'go2_navigation_runner.py'
        ).read_text(encoding='utf-8')
        self.assertIn('GO2_FIRST_PERSON_OFFSET', runner_source)
        self.assertNotIn('(0.24, 0.0, 0.08)', runner_source)

    def test_controller_uses_isaac_go2_ros2_rsl_control(self):
        controller_source = (
            _ROOT
            / 'grutopia_extension'
            / 'controllers'
            / 'go2_move_by_speed_controller.py'
        ).read_text(encoding='utf-8')
        ctrl_source = (
            _ROOT
            / 'grutopia_extension'
            / 'controllers'
            / 'isaac_go2_ctrl.py'
        ).read_text(encoding='utf-8')
        self.assertIn('Go2RSLControl', controller_source)
        self.assertIn('base_vel_cmd', ctrl_source)
        self.assertIn('POLICY_DECIMATION = 4', ctrl_source)
        self.assertIn('angular_velocity_from_quaternions', ctrl_source)
        self.assertIn('DEFAULT_ACTION_SCALE', ctrl_source)

    def test_yaw_and_body_rate_helpers_match_isaac_lab_frames(self):
        yaw_from_wxyz = _POLICY_MODULE.yaw_from_wxyz
        angular_velocity_from_quaternions = (
            _POLICY_MODULE.angular_velocity_from_quaternions
        )
        rotate_vector_inverse_wxyz = _POLICY_MODULE.rotate_vector_inverse_wxyz

        self.assertAlmostEqual(yaw_from_wxyz((1.0, 0.0, 0.0, 0.0)), 0.0, places=5)
        half = 2**0.5 / 2.0
        self.assertAlmostEqual(yaw_from_wxyz((half, 0.0, 0.0, half)), 1.5708, places=3)

        rate = angular_velocity_from_quaternions(
            (1.0, 0.0, 0.0, 0.0),
            (half, 0.0, 0.0, half),
            0.5,
        )
        self.assertGreater(rate[2], 2.5)
        self.assertLess(abs(rate[0]) + abs(rate[1]), 1e-5)

        gravity = rotate_vector_inverse_wxyz((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        self.assertTrue((abs(gravity - [0.0, 0.0, -1.0]) < 1e-5).all())


if __name__ == '__main__':
    unittest.main()
