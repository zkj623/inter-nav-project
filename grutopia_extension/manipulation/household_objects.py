"""Procedural household geometry for real-contact bimanual interaction."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


OBJECT_KINDS = ('box', 'bin', 'stool')


@dataclass(frozen=True)
class ObjectPart:
    name: str
    vertices: tuple
    color: tuple


PRISM_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def prism_vertices(outline, bottom, top):
    return tuple((float(horizontal), float(vertical), float(height))
                 for height in (bottom, top) for horizontal, vertical in outline)


def rectangular_part(name, center, dimensions, color):
    length, width, height = dimensions
    horizontal, vertical, level = center
    outline = ((horizontal - length / 2, vertical - width / 2),
               (horizontal + length / 2, vertical - width / 2),
               (horizontal + length / 2, vertical + width / 2),
               (horizontal - length / 2, vertical + width / 2))
    return ObjectPart(name, prism_vertices(outline, level - height / 2, level + height / 2), color)


def object_parts(kind, dimensions):
    if kind not in OBJECT_KINDS:
        raise ValueError(f'Unsupported household object: {kind}')
    size = np.asarray(dimensions, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.10):
        raise ValueError('Household object dimensions must be finite and exceed 0.10 m')
    length, width, height = size
    if kind == 'box':
        return [rectangular_part('Body', (0, 0, height / 2), size, (0.60, 0.39, 0.20))]
    if kind == 'stool':
        seat = min(0.06, height / 4)
        leg_width = min(0.055, min(length, width) / 6)
        parts = [rectangular_part('Seat', (0, 0, height - seat / 2), (length, width, seat), (0.56, 0.30, 0.12))]
        for index, (side_x, side_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            center = (side_x * (length / 2 - leg_width), side_y * (width / 2 - leg_width), (height - seat) / 2)
            parts.append(rectangular_part(f'Leg{index}', center, (leg_width, leg_width, height - seat), (0.39, 0.22, 0.10)))
        return parts
    wall = min(0.018, min(length, width) / 16)
    bottom = 0.02
    parts = []
    segments = 20
    for index in range(segments):
        angles = (2 * math.pi * index / segments, 2 * math.pi * (index + 1) / segments)
        outer = [(length / 2 * math.cos(angle), width / 2 * math.sin(angle)) for angle in angles]
        inner = [((length / 2 - wall) * math.cos(angle), (width / 2 - wall) * math.sin(angle)) for angle in angles]
        parts.append(ObjectPart(f'Wall{index:02d}', prism_vertices((outer[0], outer[1], inner[1], inner[0]), bottom, height),
                                (0.16, 0.39, 0.48)))
        floor_outline = ((0.0, 0.0), outer[0],
                         ((outer[0][0] + outer[1][0]) / 2, (outer[0][1] + outer[1][1]) / 2), outer[1])
        parts.append(ObjectPart(f'Floor{index:02d}', prism_vertices(floor_outline, 0, bottom), (0.13, 0.30, 0.37)))
    return parts


def grasp_height(kind, dimensions):
    height = dimensions[2]
    if kind == 'box':
        return min(0.25, height - 0.03) if height <= 0.28 else height - 0.03
    if kind == 'stool':
        # Target the upper quarter of the seat side to improve support and torso clearance.
        return height - min(0.06, height / 4) / 4
    if kind == 'bin':
        return 0.70 * height
    raise ValueError(f'Unsupported household object: {kind}')


def create_household_object(specification, destination):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    specification.validate()
    parts = object_parts(specification.kind, specification.dimensions)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/HouseholdObject').GetPrim()
    stage.SetDefaultPrim(root)
    root.CreateAttribute('rsi:object_kind', Sdf.ValueTypeNames.String).Set(specification.kind)
    root.CreateAttribute('rsi:grasp_height_m', Sdf.ValueTypeNames.Double).Set(specification.grasp_height)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(specification.mass)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(8)
    material = UsdShade.Material.Define(stage, '/HouseholdObject/ContactMaterial')
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(0.9)
    physical.CreateDynamicFrictionAttr(0.8)
    physical.CreateRestitutionAttr(0.0)
    for part in parts:
        mesh = UsdGeom.Mesh.Define(stage, '/HouseholdObject/' + part.name)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in part.vertices])
        mesh.CreateFaceVertexCountsAttr([4] * len(PRISM_FACES))
        mesh.CreateFaceVertexIndicesAttr([index for face in PRISM_FACES for index in face])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*part.color)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        collision.CreateContactOffsetAttr(0.002)
        collision.CreateRestOffsetAttr(0.0001)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    stage.GetRootLayer().Save()
    return destination
