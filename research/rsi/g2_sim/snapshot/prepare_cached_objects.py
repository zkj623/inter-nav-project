"""Reuse previously audited object convex pieces; robot-specific data is excluded."""
from pathlib import Path
import sys,json,hashlib,traceback
ROOT=Path(__file__).resolve().parent
CAT=ROOT.parent/'r1lite-household-catalog20-20261028'
sys.path[:0]=[str(ROOT.parent/'r1lite-contact-geometry-final-20261110/research/rsi/household_catalog'),str(ROOT.parent/'r1lite-collision-deps')]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':7,'physics_gpu':7,'multi_gpu':False})
try:
    import numpy as np,omni.usd
    from scipy.spatial import ConvexHull
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
    from load_asset import add_asset
    from contact_surfaces import visual_mesh
    sources=json.loads((ROOT/'object_cache_sources.json').read_text());out=ROOT/'object_assets'
    # Small objects first so rollout validation does not wait for dense chair meshes.
    for aid,src in sorted(sources.items(),key=lambda item:item[1]['hulls']):
        if (out/(aid+'.usdc')).exists():continue
        print('CACHE_OBJECT '+aid,flush=True)
        raw=Path(src['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==src['sha256']
        data=json.loads(raw);assert data['contact_representation_version']==4
        # Author off the live simulation stage: otherwise every intermediate
        # mesh edit triggers redundant asynchronous PhysX cooking.
        stage=Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        obj=add_asset(stage,aid,'/Object',mass_kg=1.,static_friction=1.,dynamic_friction=.9,package_root=CAT)
        visual=visual_mesh(stage,obj);np.savez_compressed(out/(aid+'.npz'),vertices=visual.vertices,faces=visual.faces)
        for prim in Usd.PrimRange(obj):
            if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        material=UsdShade.Material.Get(stage,'/Object/ContactMaterial')
        for i,hull in enumerate(data['hulls']):
            points=np.asarray(hull['vertices_object_m']);h=ConvexHull(points);faces=h.simplices.copy()
            tri=points[faces]
            wrong=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),h.equations[:,:3])<0
            faces[wrong]=faces[wrong][:,::-1]
            mesh=UsdGeom.Mesh.Define(stage,f'/Object/AuditedConvexPieces/Hull_{i:05d}');prim=mesh.GetPrim()
            mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in points]);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(prim);UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            c=PhysxSchema.PhysxCollisionAPI.Apply(prim);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(.00005)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
        stage.SetDefaultPrim(obj);stage.GetRootLayer().Export(str(out/(aid+'.usdc')))
        (out/(aid+'_cache_provenance.json')).write_text(json.dumps(src,indent=2))
        print('OBJECT_READY '+aid,flush=True)
except Exception:
    (ROOT/'prepare_cached_error.txt').write_text(traceback.format_exc());raise
finally:app.close()
