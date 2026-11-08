"""Bake source USD transforms into rigid-body-local collision vertices."""
import hashlib,json
import numpy as np


def bake_collision_meshes(stage,root,manifest_path):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inverse=cache.GetLocalToWorldTransform(root).GetInverse()
    records=[];inputs=[]
    for prim in list(Usd.PrimRange(root)):
        if not prim.IsA(UsdGeom.Mesh):continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
        mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        if len(points)==0:continue
        transform=np.asarray(cache.GetLocalToWorldTransform(prim)*root_inverse).T
        vertices=points@transform[:3,:3].T+transform[:3,3]
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int32)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
        if counts.sum()!=len(indices) or indices.min()<0 or indices.max()>=len(points):
            raise RuntimeError('Invalid collision mesh topology: '+str(prim.GetPath()))
        if np.linalg.det(transform[:3,:3])<0:
            indices=indices.copy();offset=0
            for count in counts:
                indices[offset:offset+count]=indices[offset:offset+count][::-1];offset+=count
        inputs.append((prim,vertices,counts,indices,mesh.GetOrientationAttr().Get()))
    if not inputs:raise RuntimeError('No visible mesh available to bake')
    material=UsdShade.Material.Get(stage,str(root.GetPath())+'/ContactMaterial')
    if not material:raise RuntimeError('Missing explicit contact material')
    paths=[]
    for i,(original,vertices,counts,indices,orientation) in enumerate(inputs):
        path=str(root.GetPath())+f'/NativeCollisionMeshes/Mesh_{i:03d}'
        mesh=UsdGeom.Mesh.Define(stage,path);prim=mesh.GetPrim()
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        mesh.CreateFaceVertexCountsAttr(counts.tolist());mesh.CreateFaceVertexIndicesAttr(indices.tolist())
        mesh.CreateOrientationAttr(orientation);mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim.SetCustomDataByKey('r1lite:bakedCollider',True)
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(.0001)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
        readback=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        error=float(abs(readback-vertices).max())
        if error>1e-7:raise RuntimeError('Baked collision vertices do not match source geometry')
        records.append(dict(source_prim=str(original.GetPath()),collision_prim=path,vertices=len(vertices),
            maximum_vertex_error_m=error,source_object_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest()))
        paths.append(path)
    manifest_path.write_text(json.dumps(dict(method='full source hierarchy baked into identity-transform object-local metre meshes',
        source_meshes_disabled_for_collision=True,maximum_vertex_error_m=max(r['maximum_vertex_error_m'] for r in records),
        meshes=records),indent=2))
    return paths
