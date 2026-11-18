from pathlib import Path
import sys,json,traceback
ROOT=Path(__file__).resolve().parent
SHARED=Path('/data20t/embodied/zkj/inter-nav-results/r1lite-contact-geometry-final-20261110/research/rsi/household_catalog')
CAT=Path('/data20t/embodied/zkj/inter-nav-results/r1lite-household-catalog20-20261028')
sys.path[:0]=[str(SHARED),'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps']
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False})
try:
    import numpy as np
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
    from contact_surfaces import visual_mesh
    import contact_surfaces
    from bake_collision_meshes import bake_collision_meshes
    from load_asset import add_asset
    from cooked_geometry import CookedGeometry
    from catalog_support import verify_catalog
    out=ROOT/'grasp_assets';out.mkdir(exist_ok=True)
    # Bounded decomposition budget. Actual loaded contact points are independently
    # checked against original triangles in every rollout; no envelope fallback.
    def configure(prim,*,finger=False,face_count=None):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
        a=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        a.CreateHullVertexLimitAttr(64);a.CreateMaxConvexHullsAttr(128)
        a.CreateVoxelResolutionAttr(250000);a.CreateErrorPercentageAttr(.02)
        a.CreateMinThicknessAttr(.0001);a.CreateShrinkWrapAttr(True)
        c=PhysxSchema.PhysxCollisionAPI.Apply(prim);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(.00005)
    contact_surfaces.configure_convex_decomposition=configure
    (out/'catalog_integrity.json').write_text(json.dumps(verify_catalog(CAT),indent=2))
    # Preserve imported official robot topology. Replace gripper colliders only
    # with separated visual-derived geometry to prevent invisible envelope grips.
    omni.usd.get_context().open_stage(str(ROOT/'smoke/scene.usda'))
    stage=omni.usd.get_context().get_stage()
    records=[]
    links=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and 'gripper_' in p.GetName()]
    for link in links:
        name=link.GetName(); print('PREP_ROBOT '+name,flush=True)
        mesh=visual_mesh(stage,link)
        np.savez_compressed(out/(name+'.npz'),vertices=mesh.vertices,faces=mesh.faces)
        mat=UsdShade.Material.Define(stage,str(link.GetPath())+'/ContactMaterial')
        phys=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());phys.CreateStaticFrictionAttr(1.);phys.CreateDynamicFrictionAttr(.9);phys.CreateRestitutionAttr(0.)
        bake_collision_meshes(stage,link,out/(name+'_bake.json'))
        geom=CookedGeometry.from_stage(stage,str(link.GetPath()),None,out/(name+'_cooked.json'))
        records.append({'name':name,'path':str(link.GetPath()),'bounds':geom.bounds.tolist(),'hulls':len(geom.parts)})
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr(True)
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateSolverPositionIterationCountAttr(32)
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateSolverVelocityIterationCountAttr(8)
    # Only parent/child gripper adjacency is excluded by native articulation defaults.
    stage.GetRootLayer().Export(str(out/'g2_contact.usdc'))
    (out/'gripper_manifest.json').write_text(json.dumps(records,indent=2))
    for rec in json.loads((CAT/'asset_manifest.json').read_text())['assets']:
        aid=rec['id'];print('PREP_OBJECT '+aid,flush=True)
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        obj=add_asset(stage,aid,'/Object',mass_kg=1.,static_friction=1.,dynamic_friction=.9,package_root=CAT)
        mesh=visual_mesh(stage,obj);np.savez_compressed(out/(aid+'.npz'),vertices=mesh.vertices,faces=mesh.faces)
        bake_collision_meshes(stage,obj,out/(aid+'_bake.json'))
        geom=CookedGeometry.from_stage(stage,'/Object',rec['dimensions_m'],out/(aid+'_cooked.json'))
        stage.SetDefaultPrim(obj)
        stage.GetRootLayer().Export(str(out/(aid+'.usdc')))
        print('OBJECT_READY '+aid+' '+str(len(geom.parts)),flush=True)
except Exception:
    (ROOT/'prepare_grasp_error.txt').write_text(traceback.format_exc());raise
finally:app.close()
