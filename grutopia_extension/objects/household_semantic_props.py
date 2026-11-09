"""Compound household props with Isaac semantic labels for open-vocabulary search."""

import numpy as np
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.semantics import add_update_semantics

from grutopia.core.scene.object import ObjectCommon, Scene
from grutopia_extension.configs.objects import HouseholdSemanticPropsCfg


@ObjectCommon.register('HouseholdSemanticProps')
class HouseholdSemanticProps(ObjectCommon):
    """Place a far refrigerator plus a chair and plant as semantic landmarks."""

    def __init__(self, config: HouseholdSemanticPropsCfg):
        super().__init__(config=config)
        self._config = config

    def set_up_scene(self, scene: Scene):
        create_prim(self._config.prim_path, prim_type='Xform')
        self._add_refrigerator(scene)
        self._add_chair(scene)
        self._add_plant(scene)

    def _add_refrigerator(self, scene: Scene):
        origin = np.asarray(self._config.refrigerator_position, dtype=float)
        self._add_part(
            scene,
            'refrigerator/body',
            'refrigerator',
            origin,
            (0.70, 0.72, 1.80),
            (0.88, 0.90, 0.93),
        )
        self._add_part(
            scene,
            'refrigerator/freezer_seam',
            'refrigerator',
            origin + np.array((0.36, 0.0, 0.35)),
            (0.04, 0.68, 0.03),
            (0.18, 0.20, 0.22),
        )
        self._add_part(
            scene,
            'refrigerator/handle',
            'refrigerator',
            origin + np.array((0.38, 0.22, -0.15)),
            (0.04, 0.04, 0.55),
            (0.12, 0.12, 0.14),
        )

    def _add_chair(self, scene: Scene):
        origin = np.asarray(self._config.chair_position, dtype=float)
        self._add_part(
            scene,
            'chair/seat',
            'chair',
            origin,
            (0.46, 0.46, 0.08),
            (0.45, 0.28, 0.16),
        )
        self._add_part(
            scene,
            'chair/back',
            'chair',
            origin + np.array((0.0, -0.19, 0.32)),
            (0.46, 0.08, 0.56),
            (0.45, 0.28, 0.16),
        )
        for index, offset in enumerate(((-0.18, -0.18), (-0.18, 0.18), (0.18, -0.18), (0.18, 0.18))):
            self._add_part(
                scene,
                f'chair/leg_{index}',
                'chair',
                origin + np.array((offset[0], offset[1], -0.26)),
                (0.06, 0.06, 0.44),
                (0.32, 0.20, 0.12),
            )

    def _add_plant(self, scene: Scene):
        origin = np.asarray(self._config.plant_position, dtype=float)
        self._add_part(
            scene,
            'plant/pot',
            'plant',
            origin + np.array((0.0, 0.0, -0.28)),
            (0.32, 0.32, 0.28),
            (0.62, 0.32, 0.18),
        )
        self._add_part(
            scene,
            'plant/foliage',
            'plant',
            origin + np.array((0.0, 0.0, 0.28)),
            (0.62, 0.62, 0.70),
            (0.18, 0.55, 0.22),
        )

    def _add_part(self, scene: Scene, child: str, label: str, position, scale, color):
        cube = FixedCuboid(
            prim_path=self.child_path(child),
            name=self.child_name(child.replace('/', '_')),
            position=np.asarray(position, dtype=float),
            scale=np.asarray(scale, dtype=float),
            color=np.asarray(color, dtype=float),
        )
        scene.add(cube)
        add_update_semantics(cube.prim, label)

    def child_path(self, child: str) -> str:
        return f'{self._config.prim_path}/{child}'

    def child_name(self, child: str) -> str:
        return f'{self._config.name}_{child}'
