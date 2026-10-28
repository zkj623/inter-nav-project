import numpy as np
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.semantics import add_update_semantics

from grutopia.core.scene.object import ObjectCommon, Scene
from grutopia_extension.configs.objects import FixedCubeCfg


@ObjectCommon.register('FixedCube')
class FixedCube(ObjectCommon):
    def __init__(self, config: FixedCubeCfg):
        super().__init__(config=config)
        self._config = config

    def set_up_scene(self, scene: Scene):
        # Contact properties must exist when PhysX creates the shape. Binding
        # them after World.reset leaves the initial contact material/patch in
        # use during startup and produced a slipping, pitching Go2 gait.
        material = None
        static = self._config.static_friction
        dynamic = self._config.dynamic_friction
        if static is not None or dynamic is not None:
            if static is None or dynamic is None or not np.isfinite([static, dynamic]).all() or min(static, dynamic) < 0:
                raise ValueError('FixedCube friction requires two finite nonnegative coefficients')
            from omni.isaac.core.materials import PhysicsMaterial
            material = PhysicsMaterial(
                prim_path=self._config.prim_path + '_ContactMaterial',
                static_friction=static, dynamic_friction=dynamic, restitution=0.0,
            )
        cube = FixedCuboid(
            prim_path=self._config.prim_path,
            name=self._config.name,
            position=np.asarray(self._config.position, dtype=float),
            orientation=np.asarray(self._config.orientation, dtype=float),
            scale=np.asarray(self._config.scale, dtype=float),
            color=np.asarray(self._config.color, dtype=float),
            physics_material=material,
        )
        if self._config.torsional_patch_radius is not None:
            cube.set_torsional_patch_radius(self._config.torsional_patch_radius)
        if self._config.min_torsional_patch_radius is not None:
            cube.set_min_torsional_patch_radius(self._config.min_torsional_patch_radius)
        scene.add(cube)
        if not self._config.visible:
            from pxr import UsdGeom
            UsdGeom.Imageable(cube.prim).MakeInvisible()
        if self._config.semantic_label:
            add_update_semantics(cube.prim, self._config.semantic_label)
