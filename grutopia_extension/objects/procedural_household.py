"""Small programmatic household scene for interaction smoke tests."""

import numpy as np
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdLux, UsdPhysics

from grutopia.core.scene.object import ObjectCommon, Scene
from grutopia_extension.configs.objects import ProceduralHouseholdCfg

OBSTACLE_CHILD = 'obstacle'
CARRY_OBJECT_CHILD = 'carry_object'
DOOR_CHILD = 'door'
DOOR_ANCHOR_CHILD = 'door_hinge_anchor'
DOOR_JOINT_CHILD = 'door_hinge_joint'


@ObjectCommon.register('ProceduralHousehold')
class ProceduralHousehold(ObjectCommon):
    """Floor, pushable box, pickup cube, and passive hinged door."""

    def __init__(self, config: ProceduralHouseholdCfg):
        super().__init__(config=config)
        self._config = config

    def set_up_scene(self, scene: Scene):
        config = self._config
        root = config.prim_path
        create_prim(root, prim_type='Xform')
        stage = get_current_stage()
        dome_light = UsdLux.DomeLight.Define(stage, f'{root}/DomeLight')
        dome_light.CreateIntensityAttr(900.0)
        distant_light = UsdLux.DistantLight.Define(stage, f'{root}/DistantLight')
        distant_light.CreateIntensityAttr(2500.0)
        distant_light.CreateAngleAttr(1.0)
        distant_light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 35.0))

        floor_scale = np.asarray(config.floor_scale, dtype=float)
        floor_position = self._world_position((floor_scale[0] / 2.0 - 1.0, 0.0, -floor_scale[2] / 2.0))
        self._add_fixed(
            scene,
            child='floor',
            position=floor_position,
            scale=floor_scale,
            color=(0.35, 0.38, 0.42),
        )

        self._add_dynamic(
            scene,
            child=OBSTACLE_CHILD,
            position=self._world_position(config.obstacle_position),
            scale=config.obstacle_scale,
            color=(0.85, 0.25, 0.18),
            mass=config.obstacle_mass,
        )

        if config.include_carry_object:
            self._add_fixed(
                scene,
                child='pedestal',
                position=self._world_position(config.pedestal_position),
                scale=config.pedestal_scale,
                color=(0.42, 0.28, 0.16),
            )
            self._add_dynamic(
                scene,
                child=CARRY_OBJECT_CHILD,
                position=self._world_position(config.carry_object_position),
                scale=config.carry_object_scale,
                color=(0.2, 0.65, 0.9),
                mass=config.carry_object_mass,
            )
            carry_prim = get_current_stage().GetPrimAtPath(self.child_path(CARRY_OBJECT_CHILD))
            UsdPhysics.RigidBodyAPI.Apply(carry_prim).CreateKinematicEnabledAttr(True)

        self._add_door(scene)
        self._add_goal_marker(scene)

    def _add_door(self, scene: Scene):
        config = self._config
        hinge = self._world_position(config.door_hinge_position)
        hinge_anchor = np.array([hinge[0], hinge[1], hinge[2] + config.door_height / 2.0])
        door_position = hinge_anchor + np.array([0.0, -config.door_width / 2.0, 0.0])
        door_scale = (config.door_thickness, config.door_width, config.door_height)

        self._add_dynamic(
            scene,
            child=DOOR_CHILD,
            position=door_position,
            scale=door_scale,
            color=(0.72, 0.48, 0.2),
            mass=config.door_mass,
        )

        frame_thickness = 0.18
        for suffix, y_position in (
            ('door_frame_hinge', hinge[1] + frame_thickness / 2.0),
            ('door_frame_latch', hinge[1] - config.door_width - frame_thickness / 2.0),
        ):
            self._add_fixed(
                scene,
                child=suffix,
                position=(hinge[0], y_position, hinge[2] + config.door_height / 2.0),
                scale=(0.22, frame_thickness, config.door_height),
                color=(0.2, 0.2, 0.22),
            )

        self._add_fixed(
            scene,
            child='door_frame_header',
            position=(hinge[0], hinge[1] - config.door_width / 2.0, hinge[2] + config.door_height + 0.1),
            scale=(0.22, config.door_width + 2.0 * frame_thickness, 0.2),
            color=(0.2, 0.2, 0.22),
        )

        stage = get_current_stage()
        anchor_prim = create_prim(
            self.child_path(DOOR_ANCHOR_CHILD),
            prim_type='Xform',
            position=hinge_anchor,
        )
        anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
        anchor_body.CreateRigidBodyEnabledAttr(False)

        joint = UsdPhysics.RevoluteJoint.Define(stage, self.child_path(DOOR_JOINT_CHILD))
        joint.CreateAxisAttr('Z')
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(config.door_open_limit_degrees)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.child_path(DOOR_ANCHOR_CHILD))])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.child_path(DOOR_CHILD))])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        # DynamicCuboid is a unit cube with non-uniform Xform scale. Joint
        # anchors are expressed in that unscaled local cube frame.
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.5, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    def _add_goal_marker(self, scene: Scene):
        config = self._config
        marker = VisualCuboid(
            prim_path=self.child_path('goal'),
            name=self.child_name('goal'),
            position=self._world_position(config.goal_position),
            scale=np.array((0.45, 0.45, 0.04)),
            color=np.array((0.15, 0.85, 0.25)),
        )
        scene.add(marker)
        add_update_semantics(marker.prim, 'goal')

    def _add_fixed(self, scene: Scene, child: str, position, scale, color):
        cube = FixedCuboid(
            prim_path=self.child_path(child),
            name=self.child_name(child),
            position=np.asarray(position, dtype=float),
            scale=np.asarray(scale, dtype=float),
            color=np.asarray(color, dtype=float),
        )
        scene.add(cube)
        add_update_semantics(cube.prim, child)

    def _add_dynamic(self, scene: Scene, child: str, position, scale, color, mass: float):
        cube = DynamicCuboid(
            prim_path=self.child_path(child),
            name=self.child_name(child),
            position=np.asarray(position, dtype=float),
            scale=np.asarray(scale, dtype=float),
            color=np.asarray(color, dtype=float),
            mass=mass,
        )
        scene.add(cube)
        add_update_semantics(cube.prim, child)

    def _world_position(self, local_position):
        return np.asarray(self._config.position, dtype=float) + np.asarray(local_position, dtype=float)

    def child_path(self, child: str) -> str:
        return f'{self._config.prim_path}/{child}'

    def child_name(self, child: str) -> str:
        return f'{self._config.name}_{child}'
