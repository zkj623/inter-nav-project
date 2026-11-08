"""Local copy of Isaac 4.2's unbound, infinite Z collision plane.

The installed SDK does not bind add_default_ground_plane's supplied material.
Preserve the effective default material rather than binding its unused 0.8 values.
"""

import numpy as np


def add_local_ground(scene):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    from omni.isaac.core.prims import GeometryPrim
    from omni.isaac.core.utils.stage import get_current_stage

    stage = get_current_stage()
    root = '/World/defaultGroundPlane'
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Xform.Define(stage, root + '/GroundPlane')
    plane = stage.DefinePrim(root + '/GroundPlane/CollisionPlane', 'Plane')
    plane.CreateAttribute('axis', Sdf.ValueTypeNames.Token, custom=False).Set('Z')
    plane.CreateAttribute('purpose', Sdf.ValueTypeNames.Token, custom=False).Set('guide')
    UsdGeom.Xformable(plane).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    UsdPhysics.CollisionAPI.Apply(plane)
    ground = GeometryPrim(str(plane.GetPath()), name='default_ground_plane',
                          position=np.zeros(3), collision=True)
    visual = UsdGeom.Mesh.Define(stage, root + '/Environment/Geometry')
    visual.CreatePointsAttr([Gf.Vec3f(-50.425, -49.575, 0), Gf.Vec3f(49.575, -49.575, 0),
                             Gf.Vec3f(-50.425, 50.425, 0), Gf.Vec3f(49.575, 50.425, 0)])
    visual.CreateFaceVertexCountsAttr([4])
    visual.CreateFaceVertexIndicesAttr([0, 1, 3, 2])
    visual.CreateSubdivisionSchemeAttr('none')
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.48, 0.55)])
    return scene.add(ground)
