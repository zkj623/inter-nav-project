"""Recover architectural floor surfaces independent of semantic class labels.

Only existing near-horizontal Base geometry within 4 cm of the reference floor
is added. No image closing, hole filling, obstacle deletion or radius change is
performed. Furniture undersides are not admitted as floor evidence.
"""
import numpy as np
import cv2
import scan

def augment_floor(stage, floor, origin, z, resolution):
    result=floor.astype(bool).copy();evidence=[]
    for prim,mesh,vertices in scan.meshes(stage):
        path=str(prim.GetPath())
        if not path.startswith('/Root/Meshes/Base/') or '/ground/' in path:continue
        if vertices[:,2].min()>z+.04 or vertices[:,2].max()<z-.04:continue
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());faces=[];offset=0
        for n in counts:
            face=ids[offset:offset+n];offset+=n
            faces.extend((face[0],face[k],face[k+1]) for k in range(1,n-1))
        if not faces:continue
        triangles=vertices[np.asarray(faces)];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);cosine=np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
        keep=(triangles[:,:,2].min(1)>=z-.04)&(triangles[:,:,2].max(1)<=z+.04)&(cosine>=.98)
        mask=np.zeros(floor.shape,np.uint8)
        for poly in np.rint((triangles[keep,:,:2]-origin)/resolution).astype(np.int32):cv2.fillConvexPoly(mask,poly,1)
        added=mask.astype(bool)&~result
        if added.any():
            evidence.append({'mesh_path':path,'added_support_m2':float(added.sum()*resolution**2),'height_range_m':[float(triangles[keep,:,2].min()),float(triangles[keep,:,2].max())]})
            result|=mask.astype(bool)
    return result,{'method':'existing architectural Base triangles; abs(height-reference)<=0.04 m; abs(normal_z)/norm>=0.98','added_support_m2':float((result&~floor.astype(bool)).sum()*resolution**2),'meshes':evidence}
