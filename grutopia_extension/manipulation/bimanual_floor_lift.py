"""Coupled IK and assets for grasping an unhandled box from both sides."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from grutopia_extension.manipulation.floor_lift import UrdfArmKinematics


@dataclass(frozen=True)
class BimanualBox:
    dimensions: tuple = (0.40, 0.50, 0.28)
    mass: float = 1.5
    grasp_height: float = 0.25
    squeeze: float = 0.003

    def validate(self):
        values = (*self.dimensions, self.mass, self.grasp_height, self.squeeze)
        if len(self.dimensions) != 3 or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('box dimensions, mass, grasp height, and squeeze must be finite and positive')
        if self.grasp_height >= self.dimensions[2]:
            raise ValueError('grasp height must lie on the box side')
        if self.squeeze >= self.dimensions[1] / 4:
            raise ValueError('squeeze is too large')


class BimanualUrdfKinematics:
    def __init__(self, urdf_path):
        self.left = UrdfArmKinematics(urdf_path, 'left', tool_offset=(0.07839, 0.0, 0.0))
        self.right = UrdfArmKinematics(urdf_path, 'right', tool_offset=(0.07839, 0.0, 0.0))
        if self.left.names[:3] != self.right.names[:3]:
            raise ValueError('both arms must share the same three torso joints')
        self.names = self.left.names + self.right.names[3:]
        self.lower = np.r_[self.left.lower, self.right.lower[3:]]
        self.upper = np.r_[self.left.upper, self.right.upper[3:]]

    def forward(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (15,):
            raise ValueError('bimanual joint vector must have 15 entries')
        return self.left.forward(positions[:9]), self.right.forward(np.r_[positions[:3], positions[9:15]])

    def solve(self, left_position, right_position, left_orientation, right_orientation, seed,
              collision_model=None, box_transform=None, box_dimensions=None, clearance=0.015, upper_limits=None,
              reseed_torso=False):
        upper = self.upper if upper_limits is None else np.minimum(self.upper, np.asarray(upper_limits, dtype=float))
        if upper.shape != self.upper.shape or not np.isfinite(upper).all() or np.any(upper - self.lower <= 2e-7):
            raise ValueError('Planning limits must be finite and leave a nonempty joint interval')
        seed = np.clip(np.asarray(seed, dtype=float), self.lower + 1e-7, upper - 1e-7)
        seeds = [seed.copy()]
        if reseed_torso:
            alternate = seed.copy()
            alternate[:3] = np.clip((-0.2, 1.5, 0.5), self.lower[:3] + 1e-7, upper[:3] - 1e-7)
            seeds.append(alternate)
        self.last_solve_diagnostics = []
        targets = (np.asarray(left_position), np.asarray(right_position))
        orientations = (np.asarray(left_orientation), np.asarray(right_orientation))

        def residual(positions):
            errors = []
            for transform, target, orientation in zip(self.forward(positions), targets, orientations):
                errors.extend(6.0 * (transform[:3, 3] - target))
                errors.extend(0.6 * Rotation.from_matrix(orientation @ transform[:3, :3].T).as_rotvec())
            errors.extend(0.01 * (positions - seed))
            if collision_model is not None:
                transforms = collision_model.forward(dict(zip(self.names, positions)))
                distances = collision_model.box_clearances(transforms, box_transform, box_dimensions)
                errors.extend(12.0 * max(0.0, clearance - value) for value in distances.values())
            return np.asarray(errors)

        for attempt, seed in enumerate(seeds):
            result = least_squares(residual, seed, bounds=(self.lower + 1e-7, upper - 1e-7),
                                   max_nfev=450, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            errors = []
            for transform, target, orientation in zip(self.forward(result.x), targets, orientations):
                distance = float(np.linalg.norm(transform[:3, 3] - target))
                angle = float(np.linalg.norm(Rotation.from_matrix(orientation @ transform[:3, :3].T).as_rotvec()))
                errors.append((distance, angle))
            accepted = not any(distance > 0.008 or angle > 0.12 for distance, angle in errors)
            self.last_solve_diagnostics.append({'attempt': attempt, 'torso_reseeded': attempt > 0,
                                                'errors_m_rad': errors, 'accepted': accepted})
            if accepted:
                return result.x
        raise RuntimeError(f'Bimanual waypoint unreachable within joint limits: {errors}')


def opposed_side_contact(hand_vectors, minimum_force=0.5):
    hand_vectors = np.asarray(hand_vectors, dtype=float)
    if hand_vectors.shape != (2, 3) or not np.isfinite(hand_vectors).all():
        return False
    return bool(hand_vectors[0, 1] > minimum_force and hand_vectors[1, 1] < -minimum_force)


def create_unhandled_box(box, destination):
    from pathlib import Path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    box.validate()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Box').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(box.mass)
    body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    body_api.CreateSolverPositionIterationCountAttr(32)
    body_api.CreateSolverVelocityIterationCountAttr(8)
    body = UsdGeom.Cube.Define(stage, '/Box/Cardboard')
    body.CreateSizeAttr(1.0)
    transform = UsdGeom.Xformable(body)
    transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, box.dimensions[2] / 2))
    transform.AddScaleOp().Set(Gf.Vec3f(*box.dimensions))
    body.CreateDisplayColorAttr([Gf.Vec3f(0.60, 0.39, 0.20)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    collision_api.CreateContactOffsetAttr(0.002)
    collision_api.CreateRestOffsetAttr(0.0001)
    material = UsdShade.Material.Define(stage, '/Box/CardboardMaterial')
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(0.9)
    physical.CreateDynamicFrictionAttr(0.8)
    physical.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material, materialPurpose='physics')
    length, width, height = box.dimensions
    for name, position, scale in (
        ('TopTape', (0, 0, height + 0.0002), (length, 0.045, 0.0004)),
        ('FrontTape', (length / 2 + 0.0002, 0, height / 2), (0.0004, 0.045, height)),
        ('BackTape', (-length / 2 - 0.0002, 0, height / 2), (0.0004, 0.045, height)),
    ):
        tape = UsdGeom.Cube.Define(stage, f'/Box/{name}')
        tape.CreateSizeAttr(1.0)
        tape_transform = UsdGeom.Xformable(tape)
        tape_transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        tape_transform.AddScaleOp().Set(Gf.Vec3f(*scale))
        tape.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.64, 0.36)])
    stage.GetRootLayer().Save()
    return destination
