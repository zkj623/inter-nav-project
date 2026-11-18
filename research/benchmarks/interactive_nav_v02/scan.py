"""Measure ground-projected area, then extract conservative navigation grids."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from pxr import Usd,UsdGeom
R=.1
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def meshes(stage):
    cache=UsdGeom.XformCache()
    for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh): continue
        m=UsdGeom.Mesh(p); v=np.asarray(m.GetPointsAttr().Get(),dtype=float)
        if v.ndim!=2 or not len(v): continue
        t=np.asarray(cache.GetLocalToWorldTransform(p));v=(v@t[:3,:3]+t[3,:3])*.01
        yield p,m,v
def floor(path):
    stage=Usd.Stage.OpenMasked(str(path),Usd.StagePopulationMask(['/Root/Meshes/Base/ground']))
    polys=[]; zs=[]
    for p,m,v in meshes(stage):
        counts=np.asarray(m.GetFaceVertexCountsAttr().Get()); ids=np.asarray(m.GetFaceVertexIndicesAttr().Get());off=0
        for n in counts:
            face=v[ids[off:off+n]];off+=n
            if len(face)>=3 and np.ptp(face[:,2])<.04 and abs(np.median(face[:,2]))<.18:
                polys.append(face[:,:2]);zs.append(float(np.median(face[:,2])))
    if not polys: raise ValueError('no ground near z=0')
    allv=np.concatenate(polys);lo=np.floor(allv.min(0)/R)*R-R;hi=allv.max(0)+R
    shape=np.ceil((hi-lo)/R).astype(int)[::-1]+1
    if np.prod(shape)>4000000: raise ValueError('oversize grid')
    grid=np.zeros(shape,np.uint8)
    for poly in polys: cv2.fillPoly(grid,[np.rint((poly-lo)/R).astype(np.int32)],1)
    return grid,lo,float(np.median(zs))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--home',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--maps',action='store_true');a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    if not a.maps:
        rows=[]
        for path in sorted((a.home/'scenes').glob('*/start_result_navigation.usd')):
            try:
                grid,lo,z=floor(path);row={'scene_id':path.parent.name,'floor_area_m2':round(float(grid.sum()*R*R),2),'extent_m':(np.array(grid.shape[::-1])*R).tolist(),'floor_z':z,'scene':str(path.relative_to(a.home))}
                np.savez_compressed(a.out/(path.parent.name+'.npz'),floor=grid,origin=lo,z=z)
                rows.append(row);print(json.dumps(row),flush=True)
            except Exception as e: print(path.parent.name,str(e),flush=True)
            save(a.out/'inventory.json',rows)
    else:
        for row in json.loads((a.out/'selected.json').read_text()):
            dest=a.out/(row['scene_id']+'-map.npz')
            if dest.exists() and 'instance_proxies_included' in np.load(dest):continue
            d=np.load(a.out/(row['scene_id']+'.npz'));g=d['floor'];lo=d['origin'];blocked=np.zeros_like(g)
            stage=Usd.Stage.Open(str(a.home/row['scene']));doors=[];count=0
            for p,m,v in meshes(stage):
                name=str(p.GetPath())
                if '/ground/' in name or '/ceiling/' in name:continue
                if v[:,2].max()<.12 or v[:,2].min()>1.15:continue
                counts=np.asarray(m.GetFaceVertexCountsAttr().Get());ids=np.asarray(m.GetFaceVertexIndicesAttr().Get());off=0;tri=[]
                if np.all(counts==3):tri=ids.reshape(-1,3)
                else:
                    for n in counts:
                        face=ids[off:off+n];off+=n
                        tri.extend((face[0],face[k],face[k+1]) for k in range(1,n-1))
                    tri=np.asarray(tri,dtype=int)
                if len(tri):
                    verts=v[tri];keep=(verts[:,:,2].max(1)>.12)&(verts[:,:,2].min(1)<1.15)
                    pixels=np.rint((verts[keep,:,:2]-lo)/R).astype(np.int32)
                    # Separate triangles avoid filling empty space between disconnected furniture/walls.
                    for polygon in pixels:cv2.fillConvexPoly(blocked,polygon,1)
                count+=1
                if '/door/' in name:doors.append(v.mean(0)[:2].tolist())
            free=g.astype(bool)&~blocked.astype(bool);dist=ndi.distance_transform_edt(free)*R
            np.savez_compressed(dest,floor=g,blocked=blocked,free=free,clearance=dist,origin=lo,z=d['z'],doors=np.array(doors),triangle_projection=True,instance_proxies_included=True)
            print(row['scene_id'],'free_m2',round(free.sum()*R*R,2),'meshes',count,flush=True)
if __name__=='__main__':main()
