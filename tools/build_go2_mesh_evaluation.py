"""Offline metric geometry; never passed to online navigator."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import cv2
from pxr import Usd,UsdGeom,UsdPhysics

BASE=Path('/data20t/embodied/zkj/inter-nav-results/go2-qualified-five-20261103')
INPUT=Path('/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs')
OUT=BASE/'mesh-inputs';OUT.mkdir(exist_ok=True)
profile=json.loads((INPUT/'scene_profile.json').read_text())
stage=Usd.Stage.Open(profile['scene_asset_path']);cache=UsdGeom.XformCache()
old=np.load(INPUT/'navigation_grid.npz');shape=old['free'].shape
origin=np.array([profile['mapping']['x_limits'][0],profile['mapping']['y_limits'][0]])
res=.025;loz=.03;hiz=.61;mask=np.zeros(shape,np.uint8)
rows=[];skipped=[];started=time.monotonic()

def clip(poly,z,above):
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ia=(a[2]>=z) if above else (a[2]<=z)
        ib=(b[2]>=z) if above else (b[2]<=z)
        if ia:result.append(a)
        if ia!=ib:result.append(a+(b-a)*((z-a[2])/(b[2]-a[2])))
    return np.asarray(result).reshape(-1,3)

def cap_fill(tri,z,target):
    selected=tri[(tri[:,:,2].min(1)<z)&(tri[:,:,2].max(1)>z)]
    if not len(selected):return 0
    points=[]
    for a,b in [(0,1),(1,2),(2,0)]:
        p,q=selected[:,a],selected[:,b];cross=(p[:,2]<z)!=(q[:,2]<z)
        t=np.zeros(len(p));np.divide(z-p[:,2],q[:,2]-p[:,2],out=t,where=cross)
        v=p[:,:2]+t[:,None]*(q[:,:2]-p[:,:2]);v[~cross]=np.nan
        points.append(v)
    v=np.stack(points,1);valid=np.isfinite(v[:,:,0])
    keep=valid.sum(1)==2
    seg=v[keep][valid[keep]].reshape(-1,2,2)
    if not len(seg):return 0
    r0=max(0,int(np.floor((seg[:,:,1].min()-origin[1])/res)))
    r1=min(shape[0],int(np.ceil((seg[:,:,1].max()-origin[1])/res)))
    odd=0
    for r in range(r0,r1):
        y=origin[1]+(r+.5)*res;a,b=seg[:,0],seg[:,1]
        cross=(a[:,1]>y)!=(b[:,1]>y)
        a,b=a[cross],b[cross]
        xs=np.sort(a[:,0]+(y-a[:,1])*(b[:,0]-a[:,0])/(b[:,1]-a[:,1]))
        # Shared triangle edges need a common geometric tolerance before parity.
        if len(xs)%2:odd+=1;continue
        for left,right in xs.reshape(-1,2):
            c0=max(0,int(np.ceil((left-origin[0])/res-.5)))
            c1=min(shape[1],int(np.ceil((right-origin[0])/res-.5)))
            if c1>c0:target[r,c0:c1]=1
    return odd

for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):continue
    path=str(prim.GetPath())
    if '/InteractiveBoxes/' in path:continue
    mesh=UsdGeom.Mesh(prim);raw=mesh.GetPointsAttr().Get()
    if raw is None or len(raw)<3:continue
    ancestor=prim;scope=None;enabled=None
    while ancestor and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.CollisionAPI):
            enabled=ancestor.GetAttribute('physics:collisionEnabled').Get() is not False
            scope=str(ancestor.GetPath());break
        ancestor=ancestor.GetParent()
    if not enabled:
        skipped.append({'path':path,'collider_scope':scope,'enabled':enabled});continue
    points=np.asarray(raw,float);matrix=np.asarray(cache.GetLocalToWorldTransform(prim))
    points=points@matrix[:3,:3]+matrix[3,:3]
    if points[:,2].min()>=hiz or points[:,2].max()<=loz:continue
    if '/ground/' in path or '/ceiling/' in path:continue
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int);indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
    if np.all(counts==3):faces=indices.reshape(-1,3)
    else:
        faces=[];offset=0
        for n in counts:
            f=indices[offset:offset+n];offset+=n
            faces.extend([f[0],f[j],f[j+1]] for j in range(1,n-1))
        faces=np.asarray(faces,int).reshape(-1,3)
    tri=points[faces];mn=tri[:,:,2].min(1);mx=tri[:,:,2].max(1)
    selected=tri[(mn<hiz)&(mx>loz)]
    before=int(mask.sum());local=np.zeros(shape,np.uint8)
    for t in selected:
        p=clip(clip(t,loz,True),hiz,False)
        if len(p)<3:continue
        pixel=np.rint((p[:,:2]-origin)/res-.5).astype(np.int32)
        cv2.fillConvexPoly(local,pixel,1)
    odds=cap_fill(tri,loz+1e-7,local)+cap_fill(tri,hiz-1e-7,local)
    mask |= local
    rows.append({'path':path,'collider_scope':scope,'triangles_in_band':len(selected),'new_cells':int(mask.sum())-before,'cap_odd_rows':odds})
    if len(rows)%100==0:print(json.dumps({'meshes':len(rows),'seconds':round(time.monotonic()-started,1),'blocked_cells':int(mask.sum())}),flush=True)
support=np.isfinite(old['ground_height'])
free=support&~mask.astype(bool)
np.savez_compressed(OUT/'navigation_grid.npz',free=free,box_free=old['box_free'],ground_height=old['ground_height'],walls=old['walls'])
for name in ['scene_profile.json','layout.json','floor_config.json','tasks_preflight.json']:
    (OUT/name).write_bytes((INPUT/name).read_bytes())
report={'method':'Actual enabled collision-scope descendant mesh triangles clipped to z=[.03,.61]m, projected to fixed 2.5cm cells; parity-filled clipped solid caps. No whole-instance AABBs. Initial boxes handled separately by evaluator.',
    'asset':profile['scene_asset_path'],'asset_sha256':hashlib.sha256(Path(profile['scene_asset_path']).read_bytes()).hexdigest(),
    'resolution_m':res,'mesh_count':len(rows),'skipped_no_enabled_collider':len(skipped),'seconds':time.monotonic()-started,
    'blocked_cells_before_boxes':int((support&~free).sum()),'support_cells':int(support.sum()),'meshes':rows,'skipped':skipped}
(OUT/'mesh_geometry_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['meshes','skipped']},indent=2))
