from typing import Optional, Tuple

from grutopia.core.config.scene import ObjectCfg


class DynamicCubeCfg(ObjectCfg):
    type: Optional[str] = 'DynamicCube'
    color: Optional[Tuple[float, float, float]] = None
    mass: Optional[float] = None
    density: Optional[float] = None
    collider: Optional[bool] = True


class FixedCubeCfg(ObjectCfg):
    type: Optional[str] = 'FixedCube'
    color: Optional[Tuple[float, float, float]] = None
    semantic_label: Optional[str] = None
    visible: bool = True


class VisualCubeCfg(ObjectCfg):
    type: Optional[str] = 'VisualCube'
    color: Optional[Tuple[float, float, float]] = None


class UsdObjCfg(ObjectCfg):
    type: Optional[str] = 'UsdObject'
    usd_path: str
    collider: Optional[bool] = True


class ProceduralHouseholdCfg(ObjectCfg):
    """Small interaction scene that does not depend on GRScenes."""

    type: Optional[str] = 'ProceduralHousehold'
    floor_scale: Tuple[float, float, float] = (9.0, 3.0, 0.2)
    obstacle_position: Tuple[float, float, float] = (1.5, 0.0, 0.3)
    obstacle_scale: Tuple[float, float, float] = (0.45, 0.6, 0.6)
    obstacle_mass: float = 2.0
    pedestal_position: Tuple[float, float, float] = (2.58, -0.74, 0.315)
    pedestal_scale: Tuple[float, float, float] = (0.2, 0.5, 0.63)
    carry_object_position: Tuple[float, float, float] = (2.58, -0.74, 0.72)
    carry_object_scale: Tuple[float, float, float] = (0.18, 0.18, 0.18)
    carry_object_mass: float = 0.15
    include_carry_object: bool = True
    door_hinge_position: Tuple[float, float, float] = (4.9, 0.6, 0.0)
    door_width: float = 1.2
    door_height: float = 1.9
    door_thickness: float = 0.06
    door_mass: float = 0.01
    door_open_limit_degrees: float = 110.0
    goal_position: Tuple[float, float, float] = (7.0, 0.0, 0.02)


class HouseholdSemanticPropsCfg(ObjectCfg):
    """Recognizable household props for open-vocabulary exploration."""

    type: Optional[str] = 'HouseholdSemanticProps'
    refrigerator_position: Tuple[float, float, float] = (12.0, -2.6, 0.90)
    chair_position: Tuple[float, float, float] = (7.2, 2.6, 0.45)
    plant_position: Tuple[float, float, float] = (9.8, 3.2, 0.55)
