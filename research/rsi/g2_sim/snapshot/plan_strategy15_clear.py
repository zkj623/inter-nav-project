"""Fixed-torso surface grasps, explicit stance search and static load screening."""
from pathlib import Path
import sys,json,argparse,time
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from g2_kinematics import transform
from ideal_constraints import Envelope,grasp_body_profiles
from low_side_profiles import lowering_body_profiles
from plan_grasps import mesh,make_candidates,bvh
import fcl
from audit_self_collision import intersections
ROOT=Path(__file__).resolve().parent
def plan(aid,support_height=0.,max_plans=12,single_only=False):
    started=time.monotonic();deadline=started+480.;env=Envelope();k=env.k;m=mesh(aid);base=transform((0,0,.04));reject=Counter();out=[];attempts=0
    profiles=[dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),np.radians((-a,2*a,-a+lean)))) for a in (60,50,40,30) for lean in (10,0,20)]
    if m.extents[2]<.55:
        deep=[dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),np.radians((-60,b,60-b+lean)))) for b in (140,150,130) for lean in (10,0,20)]
        profiles=deep+profiles
    armnames={s:[n for n,j in k.by_name.items() if f'arm_{s}_joint' in n and j['kind']=='revolute'] for s in ('l','r')}
    # An upper bound for reach; only removes provably distant targets before IK.
    parent_joint={j['child']:j for j in k.joints};reach={}
    for side in ('l','r'):
        node=f'arm_{side}_end_link';length=0.
        while node!=f'arm_{side}_link1':
            joint=parent_joint[node];length+=np.linalg.norm(joint['origin'][:3,3]);node=joint['parent']
        reach[side]=length
    seedarm={'idx22_arm_l_joint2':-.6,'idx62_arm_r_joint2':-.6,'idx24_arm_l_joint4':-1.2,'idx64_arm_r_joint4':-1.2}
    def solve(targets,q,tcp,allow_body=False):
        nonlocal attempts
        if time.monotonic()>deadline:reject['search_time_budget']+=1;return None
        fk=k.fk(q,base)
        for s,t in targets.items():
            if not allow_body and np.linalg.norm(t[:3,3]-fk[f'arm_{s}_link1'][:3,3])>reach[s]+np.linalg.norm(tcp[:3,3]):reject['outside_arm_reach_bound']+=1;return None
        body_names=['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3']
        names=(body_names if allow_body else [])+sum((armnames[s] for s in targets),[]);x0=np.array([q.get(n,0.) for n in names])
        lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);x0=np.clip(x0,lo+1e-6,hi-1e-6)
        def residual(x):
            poses=k.fk({**q,**dict(zip(names,x))},base);r=[]
            for s,t in targets.items():
                actual=poses[f'arm_{s}_end_link']@tcp
                r.extend(4*(actual[:3,3]-t[:3,3]));r.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
            body_sum=sum(dict(zip(names,x)).get(n,q.get(n,0.)) for n in body_names)
            return np.r_[r,.002*(x-x0),10*max(0.,abs(body_sum)-np.radians(18.))]
        for restart in range(3):
            if restart:
                x0=np.random.default_rng(restart).uniform(np.maximum(lo,-1.5),np.minimum(hi,1.5))
                if allow_body:x0[:len(body_names)]=np.clip([q.get(n,0.) for n in body_names],lo[:len(body_names)]+1e-6,hi[:len(body_names)]-1e-6)
            attempts+=1;fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=160,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            r=residual(fit.x)
            if not any(np.linalg.norm(r[i:i+3])>.016 or np.linalg.norm(r[i+3:i+6])>.04 for i in range(0,6*len(targets),6)):
                return {**q,**dict(zip(names,fit.x))}
        reject['arm_ik_with_fixed_torso']+=1;return None
    def path(targets,seed,tcp,mode,pos,candidate):
        grasp=solve(targets,seed,tcp,allow_body=True)
        if grasp is None:return None
        # Do not let an orientation solver silently fold the body again.
        payload=[(f'arm_{s}_end_link',1/len(targets),np.array(candidate.get('contact_centers_local',{}).get(s,tcp[:3,3]))) for s in targets]
        grasp_check=env.check(grasp,payload)
        if not grasp_check['ok']:
            key='grasp_torso_envelope' if grasp_check['torso_tilt_deg']>20. else 'grasp_motor_effort'
            reject[key]+=1
            if reject[key]<=2:print('GRASP_REJECT '+json.dumps(grasp_check),flush=True)
            return None
        pretargets={s:t.copy() for s,t in targets.items()}
        for t in pretargets.values():t[:3,3]-=.09*t[:3,2]
        pre=solve(pretargets,grasp,tcp)
        if pre is None:return None
        closed=grasp.copy()
        if mode=='bimanual_spread':
            squeeze={s:t.copy() for s,t in targets.items()}
            for t in squeeze.values():t[:3,3]+=.002*t[:3,2]
            closed=solve(squeeze,grasp,tcp)
            if closed is None:return None
        else:closed['idx31_gripper_l_inner_joint1']=0.
        last=closed;lift=[]
        for dz in np.linspace(.025,.20,8):
            desired={s:t.copy() for s,t in targets.items()}
            for t in desired.values():
                t[2,3]+=dz
                if mode=='bimanual_spread':t[:3,3]+=.002*t[:3,2]
            last=solve(desired,last,tcp)
            if last is None:return None
            lift.append(last)
        ns=set(pre)|set(grasp)
        approach=[{n:(1-u)*pre.get(n,0)+u*grasp.get(n,0) for n in ns} for u in np.linspace(0,1,7)]
        checks=[env.check(q,payload if i>=len(approach) else None) for i,q in enumerate(approach+[closed]+lift)]
        if not all(c['ok'] for c in checks):reject['trajectory_posture_or_static_effort']+=1;return None
        if any(intersections(q) for q in approach+[closed]+lift):reject['visible_self_collision']+=1;return None
        return dict(asset=aid,mode=mode,candidate=candidate,object_position=pos.tolist(),base_pose=base.tolist(),tcp=tcp.tolist(),q_pre=pre,q_grasp=grasp,q_close=closed,q_lift=lift,
            fixed_body={n:v for n,v in grasp.items() if 'body_joint' in n},support_height_m=support_height,torso_envelope={'max_tilt_deg':20,'max_body_joint_deg':180,'static_effort_fraction':.8},
            envelope_checks=checks,scope='fixed torso; arms-only IK/lift; initial stance search; no navigation or object attachment')
    # Downward-tilted wrists raise the wrist above low objects. Contact-plane
    # offsets come from actual distal meshes, independently for each hand.
    openq=-.6;fkopen=k.fk({'idx31_gripper_l_inner_joint1':openq,'idx71_gripper_r_inner_joint1':openq})
    distal={}
    for side in ('l','r'):
      inv=np.linalg.inv(fkopen[f'arm_{side}_end_link']);distal[side]=[]
      for finger in ('inner','outer'):
        name=f'gripper_{side}_{finger}_link4';fm=mesh(name);t=inv@fkopen[name];distal[side].append(fm.vertices@t[:3,:3].T+t[:3,3])
    hand_shapes={}
    for theta in (35,50,20):
      for side,normal in [('l',np.array([0.,-1,0])),('r',np.array([0.,1,0]))]:
        zz=np.cos(np.radians(theta))*normal+np.array([0,0,-np.sin(np.radians(theta))]);xx=np.array([1.,0,0]);baseR=np.column_stack([xx,np.cross(zz,xx),zz]);best=None
        for phi in np.linspace(-.35,.35,141):
          rr=baseR@Rotation.from_euler('y',phi).as_matrix();nl=rr.T@normal;centers=[];depths=[]
          for v in distal[side]:
            depth=v@nl;depths.append(depth.max());centers.append(v[depth>depth.max()-.0002].mean(0))
          error=abs(depths[0]-depths[1])
          if best is None or error<best[0]:best=(error,rr,np.array(centers))
        if best[0]<.001:hand_shapes[theta,side]=best
    fractions={'bin_lidded':(.50,.60,.40),'bin_round_pedal':(.60,.72,.45),'box_white':(.75,.60,.85),'box_reinforced':(.75,.60,.85),'case_handheld':(.65,.50,.78),'stool_small':(.15,.25,.40)}
    specs=[]
    for yaw in (0,90):
      mm=m.copy();mm.apply_transform(transform(rpy=(0,0,np.radians(yaw))))
      for frac in fractions.get(aid,(.75,.60,.45)):
       z=mm.bounds[0,2]+mm.extents[2]*frac
       for theta in (35,50,20):
        pair={};locals={};witnesses={};valid=True
        for side,which in [('l',-1),('r',0)]:
          if (theta,side) not in hand_shapes:valid=False;break
          error,rr,points=hand_shapes[theta,side];center=points.mean(0);offsets=(points-center)@rr.T;hits_pair=[]
          for off in offsets:
            hits,_,faces=mm.ray.intersects_location([[off[0],mm.bounds[0,1]-.1,z+off[2]]],[[0,1,0]],multiple_hits=True)
            if len(hits)<2:valid=False;break
            order=np.argsort(hits[:,1]);ix=order[which]
            if abs(mm.face_normals[faces[ix],1])<.60:valid=False;break
            hits_pair.append(hits[ix])
          if not valid:break
          if abs(hits_pair[0][1]-hits_pair[1][1])>.003:valid=False;break
          t=np.eye(4);t[:3,:3]=rr;t[:3,3]=np.mean(hits_pair,axis=0)-rr@center;pair[side]=t;locals[side]=center.tolist();witnesses[side]=[x.tolist() for x in hits_pair]
        if not valid:reject['tilted_four_contact_geometry']+=1;continue
        for distance in (.06,.10,.14):
          pos=np.array([.36+mm.extents[0]/2+distance,0,support_height+.002]);targets={s:t.copy() for s,t in pair.items()}
          for side,t in targets.items():
            t[:3,3]+=pos
            t[1,3]+=.0025 if side=='l' else -.0025
          specs.append((targets,pos,dict(height_fraction=frac,stance_clearance_m=distance,tilt_deg=theta,object_yaw_deg=yaw,contact_centers_local=locals,contact_points_object_rotated=witnesses)))
    # Round-robin over object regions instead of exhausting every body pose on one bad target.
    for profile_index in range(len(profiles)):
      for targets,pos,candidate in specs:
        if time.monotonic()>deadline:break
        seed={**profiles[profile_index],**seedarm,'idx31_gripper_l_inner_joint1':openq,'idx71_gripper_r_inner_joint1':openq}
        p=path(targets,seed,np.eye(4),'bimanual_spread',pos,candidate)
        if p:
          p.update(object_yaw_deg=candidate['object_yaw_deg'],hold_gripper_q=openq,strategy='four_finger_tilted_wrist_support',required_four_finger_contact=True)
          out.append(p)
        if len(out)>=max_plans:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
      if time.monotonic()>deadline:break

    return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--support-height',type=float,default=0.);a=p.parse_args()
    d=plan(a.asset,a.support_height);out=ROOT/'strategy15_clear_plans';out.mkdir(exist_ok=True);(out/(a.asset+'.json')).write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='plans'}),'plans',len(d['plans']),flush=True)
