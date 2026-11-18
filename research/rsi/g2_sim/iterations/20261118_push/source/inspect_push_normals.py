import json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import trimesh
from plan_grasps import mesh
R=Path(__file__).resolve().parent;m=mesh('chair_dining')
for variant in ['pushreach','pushsafe','pushshort','pushfront']:
 p=json.loads((R/f'strategy15_{variant}_validated_plans/chair_dining.json').read_text())['plans'][0];c=p['candidate'];rz=Rotation.from_euler('z',p['object_yaw_deg'],degrees=True).as_matrix()
 pts=np.array(list(c['contact_points'].values()));_,d,faces=trimesh.proximity.closest_point(m,pts@rz)
 print(variant,'points',pts.tolist(),'normals',(m.face_normals[faces]@rz.T).tolist())
