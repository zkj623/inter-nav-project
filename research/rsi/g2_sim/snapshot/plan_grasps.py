"""Surface-guided G2 single-gripper candidates; no object pose writes in rollout."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps')
import trimesh,fcl
from g2_kinematics import Kinematics,transform
ASSETS=ROOT/'robot_assets'
def mesh(name):
    folder=ASSETS if name.startswith('gripper_') else ROOT/'object_assets'
    a=np.load(folder/(name+'.npz'));return trimesh.Trimesh(a['vertices'],a['faces'],process=False)
def bvh(m):
    s=fcl.BVHModel();s.beginModel(len(m.vertices),len(m.faces));s.addSubModel(m.vertices,m.faces.astype(np.int32));s.endModel();return s
def make_candidates(aid):
    obj=mesh(aid);bounds=obj.bounds;dims=obj.extents;candidates=[]
    for axis in (0,1):
        other=1-axis
        for z in np.linspace(bounds[1,2]-.012,max(.025,bounds[1,2]*.35),12):
            for v in np.linspace(bounds[0,other]+.008,bounds[1,other]-.008,17):
                origin=np.zeros(3);origin[axis]=bounds[0,axis]-.1;origin[other]=v;origin[2]=z
                direction=np.eye(3)[axis]
                locations,_,faces=obj.ray.intersects_location([origin],[direction],multiple_hits=True)
                if len(locations)<2:continue
                order=np.argsort(locations[:,axis]);locations=locations[order];faces=faces[order]
                for i in range(len(locations)-1):
                    lo,hi=locations[i:i+2];width=hi[axis]-lo[axis]
                    if not .003<=width<=.075:continue
                    if min(abs(obj.face_normals[faces[i],axis]),abs(obj.face_normals[faces[i+1],axis]))<.45:continue
                    center=(lo+hi)/2
                    # End-effector +Z approaches downward, +X closes across the selected section.
                    x=direction;zz=np.array([0.,0.,-1]);y=np.cross(zz,x)
                    t=np.eye(4);t[:3,:3]=np.column_stack([x,y,zz]);t[:3,3]=center
                    score=(bounds[1,2]-z)*3+np.linalg.norm(center[:2])*.5+abs(width-.035)
                    candidates.append(dict(score=float(score),width=float(width),pose=t.tolist(),axis=axis))
    candidates.sort(key=lambda c:c['score'])
    unique=[]
    for c in candidates:
        p=np.array(c['pose'])[:3,3]
        if any(c['axis']==u['axis'] and np.linalg.norm(p-np.array(u['pose'])[:3,3])<.015 for u in unique):continue
        unique.append(c)
    return unique

def plan(aid,max_plans=3):
    k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf');obj=mesh(aid);oshape=bvh(obj)
    finger_names=[p.stem for p in ASSETS.glob('gripper_l_*.npz')]
    assert len(finger_names)==8, 'Left gripper geometry preparation incomplete'
    fingers={n:bvh(mesh(n)) for n in finger_names}
    base=transform((0,0,.04));tcp=transform((0,0,.13));qopen={'idx31_gripper_l_inner_joint1':-.785,'idx71_gripper_r_inner_joint1':-.785}
    f0=k.fk(qopen);endinv=np.linalg.inv(f0['arm_l_end_link']@tcp)
    local={n:endinv@f0[n] for n in fingers}
    rejected={};plans=[];candidates=make_candidates(aid)
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for candidate in candidates[:240]:
        target=np.array(candidate['pose']);clear=True
        for height in (0.,.025,.06,.12):
            t=target.copy();t[2,3]+=height
            for name,shape in fingers.items():
                pose=t@local[name]
                body=fcl.CollisionObject(shape,fcl.Transform(pose[:3,:3],pose[:3,3]))
                r=fcl.CollisionResult();fcl.collide(body,fcl.CollisionObject(oshape),fcl.CollisionRequest(),r)
                if r.is_collision:clear=False;break
            if not clear:break
        if not clear:reject('open_finger_visual_intersection');continue
        object_pos=np.array([max(.62,.39+obj.extents[0]/2),.10,.002])
        world_target=target.copy();world_target[:3,3]+=object_pos
        for bend in ((-.7,1.5,-.8),(-1.,2.4,-1.4),(0.,0.,0.)):
            seed={**qopen,**dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),bend)),
                  'idx21_arm_l_joint1':-.7,'idx22_arm_l_joint2':-.5,'idx24_arm_l_joint4':-1.2}
            q,pe,re=k.solve('l',world_target,seed,base,tcp,body=True)
            if pe>.004 or re>.04:reject('grasp_ik');continue
            pre=world_target.copy();pre[2,3]+=.12
            qpre,pe,re=k.solve('l',pre,q,base,tcp)
            if pe>.004 or re>.04:reject('pregrasp_ik');continue
            waypoints=[];last=q;good=True
            for dz in np.linspace(.025,.20,8):
                lift=world_target.copy();lift[2,3]+=dz
                last,pe,re=k.solve('l',lift,last,base,tcp)
                if pe>.004 or re>.04:good=False;break
                waypoints.append(last)
            if not good:reject('lift_ik');continue
            plans.append(dict(asset=aid,candidate=candidate,object_position=object_pos.tolist(),base_pose=base.tolist(),
                tcp=tcp.tolist(),q_pre=qpre,q_grasp=q,q_lift=waypoints,scope='single left gripper top approach to mesh section'))
            break
        if len(plans)>=max_plans:break
    return dict(asset=aid,candidates=len(candidates),rejections=rejected,plans=plans)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--max-plans',type=int,default=3);a=p.parse_args()
    out=ROOT/'plans';out.mkdir(exist_ok=True)
    result=plan(a.asset,a.max_plans);(out/(a.asset+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='plans'}), 'plans',len(result['plans']),flush=True)
