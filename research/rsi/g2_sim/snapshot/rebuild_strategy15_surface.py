"""Replace over-filled convex unions with bounded-thickness triangle prisms."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
R=Path(__file__).resolve().parent;aid=sys.argv[1];out=R/'strategy15_surface_assets';out.mkdir(exist_ok=True)
src=R/'object_assets'/(aid+'.usdc');stage=Usd.Stage.Open(str(src));root=stage.GetDefaultPrim();meshpath=R/'object_assets'/(aid+'.npz')
proxy=R/'strategy15_surface_proxies'/(aid+'.npz')
if proxy.exists():
 check=json.loads(proxy.with_name(aid+'_check.json').read_text());assert check['passed_sampled_check'];meshpath=proxy
d=np.load(meshpath);tri=d['vertices'][d['faces']];half=.00015;hulls=[]
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
material=UsdShade.Material.Get(stage,str(root.GetPath())+'/ContactMaterial')
for i,t in enumerate(tri):
 n=np.cross(t[1]-t[0],t[2]-t[0]);area=np.linalg.norm(n)
 if area<1e-12:continue
 n/=area;v=np.concatenate([t+n*half,t-n*half]);h=ConvexHull(v);faces=h.simplices.copy();tt=v[faces];bad=np.einsum('ij,ij->i',np.cross(tt[:,1]-tt[:,0],tt[:,2]-tt[:,0]),h.equations[:,:3])<0;faces[bad]=faces[bad][:,::-1]
 mesh=UsdGeom.Mesh.Define(stage,str(root.GetPath())+f'/TriangleSurfacePrisms/Tri_{i:05d}');p=mesh.GetPrim();mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in v]);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
 UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
 p.AddAppliedSchema('PhysxCollisionAPI');p.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0005);p.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(.00005)
 if material:UsdShade.MaterialBindingAPI.Apply(p).Bind(material,materialPurpose='physics')
 hulls.append(dict(vertices_object_m=v.tolist()))
stage.Flatten().Export(str(out/(aid+'.usdc')))
(out/(aid+'_cooked.json')).write_text(json.dumps(dict(hulls=hulls,method='triangle_prisms',half_thickness_m=half,source=str(src),source_triangles=len(tri)),indent=2))
print(aid,'triangle_prisms',len(hulls),'max normal extrusion',half,flush=True)
