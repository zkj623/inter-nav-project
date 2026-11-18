from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation
from plan_grasps import mesh
def candidates(aid):
 d=json.loads(Path(__file__).with_name('corners_geometry_'+aid+'.json').read_text());out=[];vertices=mesh(aid).vertices
 for angle in (0,-15,15,-30,30,-45,45):
  rot=Rotation.from_euler('z',angle,degrees=True).as_matrix()
  for c in d['candidates']:
   c=json.loads(json.dumps(c));c['yaw']+=angle
   for key in ['left','right']:
    t=np.array(c[key]['pose']);t[:3,:3]=rot@t[:3,:3];t[:3,3]=rot@t[:3,3];c[key]['pose']=t.tolist()
    c[key]['contacts']=(np.array(c[key]['contacts'])@rot.T).tolist()
   r=Rotation.from_euler('z',c['yaw'],degrees=True).as_matrix();c['extent_x']=float(np.ptp(vertices@r[0,:]))
   out.append(c)
 return out,{'source_candidates':len(d['candidates']),'yaw_offsets':[0,-15,15,-30,30,-45,45]}
