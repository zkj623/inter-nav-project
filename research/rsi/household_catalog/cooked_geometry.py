"""Queries against the SAME convex pieces cooked by PhysX, in object metres."""
import hashlib,json
import numpy as np
from scipy.spatial import ConvexHull
import fcl


def transform_points(points, transform):
    points=np.asarray(points,dtype=float); transform=np.asarray(transform,dtype=float)
    return points@transform[:3,:3].T+transform[:3,3]


def metric_bounds_diagnostics(bounds,expected_dimensions):
    bounds=np.asarray(bounds);expected=np.asarray(expected_dimensions)
    sizes=bounds[1]-bounds[0]
    delta=sizes-expected
    # This catches unit/transform mistakes, not allowed contact penetration.
    # Cooking can expand source bounds by several mm; queries must keep that shape.
    valid=bool(np.isfinite(bounds).all() and np.all(expected>0) and
        np.all(abs(delta)<=np.maximum(.01,.05*expected)) and
        np.max(abs(bounds.mean(axis=0)-np.array([0.,0.,expected[2]/2])))<=.01)
    return dict(valid_metric_transform=valid,visual_dimensions_m=expected.tolist(),
                cooked_dimensions_m=sizes.tolist(),cooking_dimension_delta_m=delta.tolist(),
                cooking_expansion_above_3mm=bool(np.max(abs(delta))>.003))


def grasp_target_errors(actual,desired,normal,normal_tolerance,tangential_tolerance=.001):
    delta=np.asarray(actual)-np.asarray(desired);normal=np.asarray(normal)
    signed=delta@normal
    tangential=np.linalg.norm(delta-signed[:,None]*normal,axis=1)
    return dict(normal_errors_m=signed.tolist(),tangential_errors_m=tangential.tolist(),
        normal_tolerance_m=normal_tolerance,tangential_tolerance_m=tangential_tolerance,
        passed=bool(np.isfinite(delta).all() and np.max(abs(signed))<=normal_tolerance
                    and np.max(tangential)<=tangential_tolerance))


class ConvexPart:
    def __init__(self, points, name):
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or not np.isfinite(self.points).all():
            raise ValueError('Invalid cooked vertices')
        hull=ConvexHull(self.points)
        # FCL requires consistent outward polygon winding.
        faces=hull.simplices.copy()
        for i,face in enumerate(faces):
            p,q,r=self.points[face]
            if np.dot(np.cross(q-p,r-p),hull.equations[i,:3])<0:faces[i]=face[::-1]
        polygons=np.column_stack((np.full(len(faces),3),faces)).astype(np.int32).ravel()
        self.shape=fcl.Convex(self.points,len(faces),polygons)
        self.object=fcl.CollisionObject(self.shape)
        self.equations=hull.equations.copy()
        self.bounds=np.array([self.points.min(0),self.points.max(0)])
        self.name=name

    def ray_y(self,x,z):
        n=self.equations[:,:3]; b=self.equations[:,3]
        rhs=-(n[:,0]*x+n[:,2]*z+b)
        if np.any((abs(n[:,1])<1e-10)&(rhs < -1e-9)):return None
        pos=n[:,1]>1e-10;neg=n[:,1]<-1e-10
        if not pos.any() or not neg.any():return None
        hi_indices=np.where(pos)[0];lo_indices=np.where(neg)[0]
        hi_values=rhs[pos]/n[pos,1];lo_values=rhs[neg]/n[neg,1]
        hi_i=hi_indices[np.argmin(hi_values)];lo_i=lo_indices[np.argmax(lo_values)]
        lo=rhs[lo_i]/n[lo_i,1];hi=rhs[hi_i]/n[hi_i,1]
        if lo>hi+1e-9:return None
        return float(lo),float(hi),n[lo_i].copy(),n[hi_i].copy()


class CookedGeometry:
    def __init__(self, parts):
        self.parts=list(parts)
        if not self.parts:raise RuntimeError('No cooked convex shapes; refusing bounding-box fallback')
        self.points=np.concatenate([p.points for p in self.parts])
        self.bounds=np.array([self.points.min(0),self.points.max(0)])
        self.request=fcl.DistanceRequest(enable_signed_distance=True)

    def gap(self, shape, vertices, local_pose):
        """Signed convex-union gap; AABB only prunes provably farther pieces."""
        pts=transform_points(vertices,local_pose)
        lower,upper=pts.min(0),pts.max(0)
        bounds=np.array([p.bounds for p in self.parts])
        disjoint=np.maximum(np.maximum(bounds[:,0]-upper,lower-bounds[:,1]),0.)
        lower_bound=np.linalg.norm(disjoint,axis=1)
        finger=fcl.CollisionObject(shape,fcl.Transform(local_pose[:3,:3],local_pose[:3,3]))
        best=float('inf')
        for i in np.argsort(lower_bound):
            if lower_bound[i]>max(0.,best)+1e-10:break
            distance=float(fcl.distance(finger,self.parts[i].object,self.request,fcl.DistanceResult()))
            if not np.isfinite(distance):raise RuntimeError('Non-finite native convex gap')
            best=min(best,distance)
        return best

    def section(self,x,z):
        hits=[h for part in self.parts if (h:=part.ray_y(x,z)) is not None]
        if not hits:return None
        left=max(hits,key=lambda h:h[1]);right=min(hits,key=lambda h:h[0])
        return dict(right=right[0],left=left[1],right_normal=right[2],left_normal=left[3])

    def grasp_profile(self,preferred_height):
        length,width,height=self.bounds[1]-self.bounds[0]
        candidates=[]
        # Test finite contact patches rather than an isolated ray/handle tip.
        for z in dict.fromkeys([float(preferred_height),float(height*.7),float(height*.5),float(height*.85)]):
            for x in (0.,-.10*length,.10*length):
                hits=[self.section(x+dx,z+dz) for dx,dz in [(0.,0.),(-.008,0.),(.008,0.),(0.,-.008),(0.,.008)]]
                if any(h is None for h in hits):continue
                center=hits[0];span=center['left']-center['right']
                if not .14<=span<=.65:continue
                if min(h['left_normal'][1] for h in hits)<.94 or max(h['right_normal'][1] for h in hits)>-.94:continue
                variation=max(max(abs(h['left']-center['left']),abs(h['right']-center['right'])) for h in hits)
                if variation>.004:continue
                score=abs(z-preferred_height)+abs(x)*.5+variation*4
                candidates.append((score,dict(x_m=x,height_m=z,center_y_m=(center['left']+center['right'])/2,
                    half_width_m=span/2,left_surface_m=center['left'],right_surface_m=center['right'],
                    patch_variation_m=variation,patch_radius_m=.008)))
        if not candidates:raise RuntimeError('NO_SUPPORTED_BILATERAL_SIDE_PATCH_ON_COOKED_GEOMETRY')
        return min(candidates,key=lambda x:x[0])[1]

    @classmethod
    def from_stage(cls,stage,root_path,expected_dimensions,output):
        from omni.physx import get_physx_cooking_interface
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        cooking=get_physx_cooking_interface();root=stage.GetPrimAtPath(root_path)
        cache=UsdGeom.XformCache(Usd.TimeCode.Default());parts=[];records=[]
        diagnostics_meshes=[]
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible and not prim.GetCustomDataByKey('r1lite:bakedCollider'):continue
            path=str(prim.GetPath());returned=[]
            def receive(status,convexes):returned.append((status,convexes))
            cooking.request_convex_collision_representation(
                stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False,on_result=receive)
            if not returned or returned[0][0]!=PhysxCollisionRepresentationResult.RESULT_VALID:
                raise RuntimeError('COOKED_DATA_MISSING: '+path)
            convexes=returned[0][1]
            if not convexes:raise RuntimeError('COOKED_DATA_EMPTY: '+path)
            matrix=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(root).GetInverse()
            transform=np.asarray(matrix,dtype=float).T
            source_points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            raw_points=transform_points(source_points,transform)
            cooked_points=np.concatenate([np.array([[p.x,p.y,p.z] for p in data.vertices]) for data in convexes])
            diagnostics_meshes.append(dict(prim_path=path,visible_bounds_object_m=[raw_points.min(0).tolist(),raw_points.max(0).tolist()],
                source_local_bounds=[source_points.min(0).tolist(),source_points.max(0).tolist()],
                cooked_local_bounds=[cooked_points.min(0).tolist(),cooked_points.max(0).tolist()],mesh_to_object=transform.tolist()))
            for i,data in enumerate(convexes):
                points=transform_points(np.array([[p.x,p.y,p.z] for p in data.vertices]),transform)
                part=ConvexPart(points,f'{path}:{i}');parts.append(part)
                records.append(dict(prim_path=path,hull_index=i,vertices_object_m=points.tolist()))
        geometry=cls(parts)
        sizes=geometry.bounds[1]-geometry.bounds[0]
        diagnostics=metric_bounds_diagnostics(geometry.bounds,expected_dimensions)
        document=dict(method='PhysX request_convex_collision_representation synchronous; each hull rebased through full USD hierarchy into object metres',
            bounds_m=geometry.bounds.tolist(),dimensions_m=sizes.tolist(),hull_count=len(parts),hulls=records,
            transform_validation=diagnostics,mesh_transform_checks=diagnostics_meshes)
        text=json.dumps(document,indent=2);output.write_text(text)
        if not diagnostics['valid_metric_transform']:
            raise RuntimeError('COOKED_UNITS_OR_TRANSFORM_MISMATCH: '+str(diagnostics))
        geometry.sha256=hashlib.sha256(text.encode()).hexdigest()
        return geometry
