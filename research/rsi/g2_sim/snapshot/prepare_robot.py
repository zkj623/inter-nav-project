"""Visual-derived per-link convex decomposition, with actual-contact audit downstream."""
from pathlib import Path
import sys,json,traceback
ROOT=Path(__file__).resolve().parent
sys.path[:0]=['/data20t/embodied/zkj/inter-nav-results/r1lite-contact-geometry-final-20261110/research/rsi/household_catalog','/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps']
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False})
try:
    import numpy as np,omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
    from contact_surfaces import visual_mesh
    from cooked_geometry import CookedGeometry
    out=ROOT/'robot_assets';out.mkdir(exist_ok=True)
    omni.usd.get_context().open_stage(str(ROOT/'smoke/scene.usda'));stage=omni.usd.get_context().get_stage()
    links=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and 'gripper_' in p.GetName()]
    records=[]
    for link in links:
        name=link.GetName();print('ROBOT_LINK '+name,flush=True)
        source=visual_mesh(stage,link);np.savez_compressed(out/(name+'.npz'),vertices=source.vertices,faces=source.faces)
        for child in Usd.PrimRange(link):
            if child.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        mesh=UsdGeom.Mesh.Define(stage,str(link.GetPath())+'/VisualContactMesh')
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in source.vertices]);mesh.CreateFaceVertexCountsAttr([3]*len(source.faces));mesh.CreateFaceVertexIndicesAttr(source.faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible');p=mesh.GetPrim()
        p.SetCustomDataByKey('r1lite:bakedCollider',True)
        UsdPhysics.CollisionAPI.Apply(p);UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexDecomposition')
        a=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p);a.CreateHullVertexLimitAttr(64);a.CreateMaxConvexHullsAttr(128);a.CreateVoxelResolutionAttr(500000);a.CreateShrinkWrapAttr(True);a.CreateErrorPercentageAttr(.02);a.CreateMinThicknessAttr(.0001)
        c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(.00005)
        mat=UsdShade.Material.Define(stage,str(link.GetPath())+'/ContactMaterial');ph=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());ph.CreateStaticFrictionAttr(1.);ph.CreateDynamicFrictionAttr(.9);ph.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
        geom=CookedGeometry.from_stage(stage,str(link.GetPath()),None,out/(name+'_cooked.json'))
        records.append(dict(name=name,path=str(link.GetPath()),hulls=len(geom.parts),method='per-link visible triangle mesh convex decomposition; 3mm contact-point witness required'))
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            a=PhysxSchema.PhysxArticulationAPI.Apply(p);a.CreateEnabledSelfCollisionsAttr(True);a.CreateSolverPositionIterationCountAttr(32);a.CreateSolverVelocityIterationCountAttr(8)
    stage.GetRootLayer().Export(str(out/'g2_contact.usdc'))
    (out/'gripper_manifest.json').write_text(json.dumps(records,indent=2));print('ROBOT_READY',flush=True)
except Exception:
    (ROOT/'prepare_robot_error.txt').write_text(traceback.format_exc());raise
finally:app.close()
