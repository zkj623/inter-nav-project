"""Independent object preparation; no robot processes or artifact mutation."""
from pathlib import Path
import sys,json,traceback
ROOT=Path(__file__).resolve().parent
SHARED=Path('/data20t/embodied/zkj/inter-nav-results/r1lite-contact-geometry-final-20261110/research/rsi/household_catalog')
CAT=Path('/data20t/embodied/zkj/inter-nav-results/r1lite-household-catalog20-20261028')
sys.path[:0]=[str(SHARED),'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps']
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':7,'physics_gpu':7,'multi_gpu':False})
try:
    import numpy as np,omni.usd
    from pxr import UsdGeom,UsdPhysics,PhysxSchema
    import contact_surfaces
    from contact_surfaces import visual_mesh
    from bake_collision_meshes import bake_collision_meshes
    from load_asset import add_asset
    from cooked_geometry import CookedGeometry
    out=ROOT/'object_assets';out.mkdir(exist_ok=True)
    def configure(prim,*,finger=False,face_count=None):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
        a=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        a.CreateHullVertexLimitAttr(64);a.CreateMaxConvexHullsAttr(128)
        a.CreateVoxelResolutionAttr(250000);a.CreateErrorPercentageAttr(.02)
        a.CreateMinThicknessAttr(.0001);a.CreateShrinkWrapAttr(True)
        c=PhysxSchema.PhysxCollisionAPI.Apply(prim);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(.00005)
    contact_surfaces.configure_convex_decomposition=configure
    for rec in json.loads((CAT/'asset_manifest.json').read_text())['assets']:
        aid=rec['id'];print('PREP_OBJECT '+aid,flush=True)
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        obj=add_asset(stage,aid,'/Object',mass_kg=1.,static_friction=1.,dynamic_friction=.9,package_root=CAT)
        mesh=visual_mesh(stage,obj);np.savez_compressed(out/(aid+'.npz'),vertices=mesh.vertices,faces=mesh.faces)
        bake_collision_meshes(stage,obj,out/(aid+'_bake.json'))
        geom=CookedGeometry.from_stage(stage,'/Object',rec['dimensions_m'],out/(aid+'_cooked.json'))
        stage.SetDefaultPrim(obj);stage.GetRootLayer().Export(str(out/(aid+'.usdc')))
        print('OBJECT_READY '+aid+' '+str(len(geom.parts)),flush=True)
except Exception:
    (ROOT/'prepare_objects_error.txt').write_text(traceback.format_exc());raise
finally:app.close()
