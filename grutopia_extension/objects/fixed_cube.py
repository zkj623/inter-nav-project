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
        cube = FixedCuboid(
            prim_path=self._config.prim_path,
            name=self._config.name,
            position=np.asarray(self._config.position, dtype=float),
            orientation=np.asarray(self._config.orientation, dtype=float),
            scale=np.asarray(self._config.scale, dtype=float),
            color=np.asarray(self._config.color, dtype=float),
        )
        scene.add(cube)
        if not self._config.visible:
            from pxr import UsdGeom
            UsdGeom.Imageable(cube.prim).MakeInvisible()
        if self._config.semantic_label:
            add_update_semantics(cube.prim, self._config.semantic_label)
