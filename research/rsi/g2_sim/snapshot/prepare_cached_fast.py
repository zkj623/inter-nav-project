"""Batch-write dense convex layers to avoid quadratic live USD authoring."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;CAT=ROOT.parent/'r1lite-household-catalog20-20261028'
sys.path[:0]=[str(ROOT.parent/'r1lite-contact-geometry-final-20261110/research/rsi/household_catalog'),str(ROOT.parent/'r1lite-collision-deps')]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':7,'physics_gpu':7,'multi_gpu':False})
try:
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    from load_asset import add_asset
    from contact_surfaces import visual_mesh
    sources=json.loads((ROOT/'object_cache_sources.json').read_text());out=ROOT/'object_assets'
    for aid,src in sorted(sources.items(),key=lambda item:item[1]['hulls']):
        if (out/(aid+'.usdc')).exists():continue
        print('CACHE_OBJECT '+aid,flush=True)
        raw=Path(src['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==src['sha256'];data=json.loads(raw)
        textpath=out/(aid+'_hulls.usda');binary=out/(aid+'_hulls.usdc')
        with textpath.open('w') as f:
            f.write('#usda 1.0\n(defaultPrim = "Hulls")\ndef Xform "Hulls" {\n')
            for i,h in enumerate(data['hulls']):
                pts=np.asarray(h['vertices_object_m']);ch=ConvexHull(pts);faces=ch.simplices.copy()
                tri=pts[faces];wrong=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),ch.equations[:,:3])<0;faces[wrong]=faces[wrong][:,::-1]
                ps=', '.join('('+','.join(format(float(x),'.9g') for x in p)+')' for p in pts)
                fs=', '.join(map(str,faces.ravel()));counts=', '.join(['3']*len(faces))
                f.write(f'def Mesh "Hull_{i:05d}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]) {{\n')
                f.write(f'point3f[] points = [{ps}]\nint[] faceVertexCounts = [{counts}]\nint[] faceVertexIndices = [{fs}]\n')
                f.write('uniform token subdivisionScheme = "none"\ntoken visibility = "invisible"\nuniform token physics:approximation = "convexHull"\nbool physics:collisionEnabled = true\nfloat physxCollision:contactOffset = 0.0005\nfloat physxCollision:restOffset = 0.00005\n}\n')
                if i%5000==0:print(aid,i,flush=True)
            f.write('}\n')
        hullstage=Usd.Stage.Open(str(textpath));hullstage.GetRootLayer().Export(str(binary));del hullstage
        stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        obj=add_asset(stage,aid,'/Object',mass_kg=1.,static_friction=1.,dynamic_friction=.9,package_root=CAT)
        visual=visual_mesh(stage,obj);np.savez_compressed(out/(aid+'.npz'),vertices=visual.vertices,faces=visual.faces)
        for prim in Usd.PrimRange(obj):
            if prim.HasAPI(UsdPhysics.CollisionAPI):prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        hp=UsdGeom.Xform.Define(stage,'/Object/AuditedConvexPieces').GetPrim();hp.GetReferences().AddReference(str(binary))
        UsdShade.MaterialBindingAPI.Apply(hp).Bind(UsdShade.Material.Get(stage,'/Object/ContactMaterial'),materialPurpose='physics')
        stage.SetDefaultPrim(obj);stage.GetRootLayer().Export(str(out/(aid+'.usdc')))
        (out/(aid+'_cache_provenance.json')).write_text(json.dumps(src,indent=2));print('OBJECT_READY '+aid,flush=True)
finally:app.close()
