from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;AS=ROOT/'robot_assets'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':3,'physics_gpu':3,'multi_gpu':False})
try:
    import numpy as np,omni.usd
    from scipy.spatial import ConvexHull
    from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
    omni.usd.get_context().open_stage(str(AS/'g2_contact_full.usdc'));stage=omni.usd.get_context().get_stage()
    records=json.loads((AS/'all_link_manifest.json').read_text());total=0
    for rec in records:
        name=rec['name'];path=rec['path'];data=json.loads((AS/(name+'_cooked.json')).read_text())
        src=stage.GetPrimAtPath(path+'/VisualContactMesh');UsdPhysics.CollisionAPI(src).CreateCollisionEnabledAttr(False)
        mat=UsdShade.Material.Get(stage,path+'/ContactMaterial')
        for i,hull in enumerate(data['hulls']):
            pts=np.array(hull['vertices_object_m']);ch=ConvexHull(pts);faces=ch.simplices.copy()
            tri=pts[faces];wrong=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),ch.equations[:,:3])<0;faces[wrong]=faces[wrong][:,::-1]
            mesh=UsdGeom.Mesh.Define(stage,path+f'/AuditedConvexPieces/Hull_{i:03d}')
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts]);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateVisibilityAttr('invisible');mesh.CreateSubdivisionSchemeAttr('none')
            p=mesh.GetPrim();UsdPhysics.CollisionAPI.Apply(p);UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
            c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(.00005)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics');total+=1
        print('FROZEN '+name,flush=True)
    stage.GetRootLayer().Export(str(AS/'g2_contact_explicit.usdc'))
    (AS/'frozen_colliders.json').write_text(json.dumps({'hulls':total,'source':'per-link PhysX cooked convex representation; reauthored as individual hull meshes'},indent=2))
finally:app.close()
