"""Robot-independent multiband scene rasterization for connectivity screening."""
import argparse,json
from pathlib import Path
import numpy as np,cv2
from scipy import ndimage as ndi
from pxr import Usd
import scan
ap=argparse.ArgumentParser();ap.add_argument('--home',type=Path,required=True);ap.add_argument('--selected',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
scan.R=.025
bands=[(.12,.4),(.4,.8),(.8,1.15),(1.15,1.8)]
for row in json.loads(a.selected.read_text()):
 target=a.out/(row['id']+'.npz')
 if target.exists():continue
 path=a.home/row['scene'];floor,origin,z=scan.floor(path);layers=np.zeros((len(bands),*floor.shape),np.uint8);stage=Usd.Stage.Open(str(path))
 for prim,m,v in scan.meshes(stage):
  name=str(prim.GetPath())
  if '/ground/' in name or '/ceiling/' in name:continue
  if v[:,2].max()<bands[0][0] or v[:,2].min()>bands[-1][1]:continue
  counts=np.asarray(m.GetFaceVertexCountsAttr().Get());ids=np.asarray(m.GetFaceVertexIndicesAttr().Get());off=0
  if np.all(counts==3):faces=ids.reshape(-1,3)
  else:
   faces=[]
   for n in counts:
    p=ids[off:off+n];off+=n;faces.extend((p[0],p[k],p[k+1]) for k in range(1,n-1))
   faces=np.asarray(faces,dtype=int)
  if not len(faces):continue
  triangles=v[faces];lo=triangles[:,:,2].min(1);hi=triangles[:,:,2].max(1)
  for i,(zlo,zhi) in enumerate(bands):
   polys=np.rint((triangles[(hi>zlo)&(lo<zhi),:,:2]-origin)/scan.R).astype(np.int32)
   for polygon in polys:cv2.fillConvexPoly(layers[i],polygon,1)
 np.savez_compressed(target,floor=floor,obstacle_bands=layers,band_limits=bands,origin=origin,z=z,resolution=scan.R)
 print(row['id'],floor.shape,'complete',flush=True)
