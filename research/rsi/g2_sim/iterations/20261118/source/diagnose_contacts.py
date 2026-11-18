from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from plan_grasps import mesh,bvh
from g2_kinematics import Kinematics
import fcl
R=Path(__file__).resolve().parent;old=R.parent/'g2-migration-20261111'
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
summary=json.loads((old/'strategy15_summary.json').read_text())
for o in summary['objects']:
    if o['asset'] not in ['box_white','box_reinforced','case_handheld']:continue
    path=old/o['best']['path'];d=json.loads(path.read_text());ss=json.loads(path.with_name('samples.json').read_text())
    s=next(x for x in reversed(ss) if x['phase'] in ['hold','contact_refine'])
    base=np.array(d['plan']['base_pose']);base[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],base)
    op=np.array(s['object_position']);oq=np.array(s['object_quaternion']);rot=Rotation.from_quat(oq[[1,2,3,0]]).as_matrix()
    ob=fcl.CollisionObject(bvh(mesh(o['asset'])),fcl.Transform(rot,op))
    out={}
    for side in ['l','r']:
        for finger in ['inner','outer']:
            n=f'gripper_{side}_{finger}_link4';pose=fk[n]
            fb=fcl.CollisionObject(bvh(mesh(n)),fcl.Transform(pose[:3,:3],pose[:3,3]))
            gap=fcl.distance(fb,ob,fcl.DistanceRequest(),fcl.DistanceResult())
            out[n]={'gap_mm':round(gap*1000,2),'force_n':round(s['finger_forces'][n],2)}
    print(o['asset'],json.dumps(out),flush=True)
