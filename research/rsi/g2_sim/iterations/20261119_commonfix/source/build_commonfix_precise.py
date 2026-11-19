from pathlib import Path
import ast
R=Path(__file__).resolve().parent
patch="""    # Local precision experiment: rebuild only four distal collision surfaces.
    precise_names=[f'gripper_{side}_{finger}_link4' for side in ('l','r') for finger in ('inner','outer')]
    for n in precise_names:
        link=next(p for p in stage.Traverse() if p.GetName()==n and p.HasAPI(UsdPhysics.RigidBodyAPI))
        for child in list(Usd.PrimRange(link)):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        data=np.load(assets/(n+'.npz'))
        mesh=UsdGeom.Mesh.Define(stage,str(link.GetPath())+'/PrecisionCollision')
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in data['vertices']])
        mesh.CreateFaceVertexCountsAttr([3]*len(data['faces']));mesh.CreateFaceVertexIndicesAttr(data['faces'].reshape(-1).tolist())
        UsdGeom.Imageable(mesh).CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition')
        cooking=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
        cooking.CreateHullVertexLimitAttr(64);cooking.CreateMaxConvexHullsAttr(384)
        cooking.CreateVoxelResolutionAttr(2000000);cooking.CreateErrorPercentageAttr(.1);cooking.CreateShrinkWrapAttr(True);cooking.CreateMinThicknessAttr(.0001)
        collision=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());collision.CreateContactOffsetAttr(.0005);collision.CreateRestOffsetAttr(0.)
    result['precision_collision_experiment']='four distal finger visual meshes, convex decomposition max384 hulls, 2M voxels, 0.1% error, shrinkwrap; contact offset0.5mm; only prepared_scene in trial output modified; shared assets unchanged'
"""
for a in ['basket_rectangular','case_handheld']:
 src=R/f'commonfix_{a}_trial.py';s=src.read_text().replace("ROOT/'commonfix_trials'","ROOT/'commonfix_precise_trials'")
 needle="    qpre=plan['q_pre'] if plan else"
 idx=s.index(needle);s=s[:idx]+patch+s[idx:]
 ast.parse(s);(R/f'commonfix_precise_{a}_trial.py').write_text(s)
