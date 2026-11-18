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
from strategy15_frame_candidates import make_candidates
import fcl
from audit_self_collision import intersections
ROOT=Path(__file__).resolve().parent
def plan(aid,support_height=0.,max_plans=4,single_only=False):
    started=time.monotonic();deadline=started+240.;env=Envelope();k=env.k;m=mesh(aid);base=transform((0,0,.04));reject=Counter();out=[];attempts=0
    profiles=[dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),np.radians((-a,2*a,-a+lean)))) for a in (30,40,50,60,10,0) for lean in (10,0,20)]
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
    avoid_plane=999.
    bodycorners={}
    import itertools
    for bn in ('body_link2','body_link3','body_link4','body_link5'):
        cd=json.loads((ROOT/'robot_assets'/(bn+'_cooked.json')).read_text());vv=np.concatenate([np.array(h['vertices_object_m']) for h in cd['hulls']]);bb=np.array([vv.min(0),vv.max(0)]);bodycorners[bn]=np.array(list(itertools.product(*zip(bb[0],bb[1]))))
    seedarm={'idx22_arm_l_joint2':-.6,'idx62_arm_r_joint2':-.6,'idx24_arm_l_joint4':-1.2,'idx64_arm_r_joint4':-1.2}
    def solve(targets,q,tcp,allow_body=False):
        nonlocal attempts
        if time.monotonic()>deadline:reject['search_time_budget']+=1;return None
        fk=k.fk(q,base)
        for s,t in targets.items():
            if not allow_body and np.linalg.norm(t[:3,3]-fk[f'arm_{s}_link1'][:3,3])>reach[s]+np.linalg.norm(tcp[:3,3]):reject['outside_arm_reach_bound']+=1;return None
        body_names=['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3','idx04_body_joint4','idx05_body_joint5']
        names=(body_names if allow_body else [])+sum((armnames[s] for s in targets),[]);x0=np.array([q.get(n,0.) for n in names])
        lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);x0=np.clip(x0,lo+1e-6,hi-1e-6)
        if allow_body:
            lo[3:5]=[-.30,-.8];hi[3:5]=[.30,.8];x0=np.clip(x0,lo+1e-6,hi-1e-6)
        def residual(x):
            poses=k.fk({**q,**dict(zip(names,x))},base);r=[]
            for s,t in targets.items():
                actual=poses[f'arm_{s}_end_link']@tcp
                r.extend(4*(actual[:3,3]-t[:3,3]));r.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
            body_sum=float(np.arccos(np.clip((poses['body_link5'][:3,:3]@env.upright.T)[2,2],-1,1)))
            clearances=[8*max(0.,np.max(corners@poses[bn][0,:3]+poses[bn][0,3])-avoid_plane) for bn,corners in bodycorners.items()]
            return np.r_[r,.002*(x-x0),10*max(0.,abs(body_sum)-np.radians(18.)),clearances]
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
        nonlocal avoid_plane
        avoid_plane=999.  # Exact cooked geometry replaces the conservative whole-body X plane
        grasp=solve(targets,seed,tcp,allow_body=True)
        if grasp is None:return None
        # Do not let an orientation solver silently fold the body again.
        payload=[(f'arm_{s}_end_link',1/len(targets),tcp[:3,3]) for s in targets]
        if not env.check(grasp,payload)['ok']:reject['grasp_static_effort']+=1;return None
        pretargets={s:t.copy() for s,t in targets.items()}
        for t in pretargets.values():
            horizontal=t[:2,2]/max(1e-9,np.linalg.norm(t[:2,2]))
            t[:2,3]-=.09*horizontal
            t[2,3]-=.025
        pre=solve(pretargets,grasp,tcp)
        if pre is None:return None
        closed=grasp.copy()
        if mode=='bimanual_side':
            squeeze={s:t.copy() for s,t in targets.items()}
            for t in squeeze.values():t[:3,3]+=.002*t[:3,2]
            closed=solve(squeeze,grasp,tcp)
            if closed is None:return None
        elif mode!='bimanual_under':
            closed['idx31_gripper_l_inner_joint1']=0.
            closed['idx71_gripper_r_inner_joint1']=0.
        last=closed;lift=[]
        for dz in np.linspace(.025,.20,8):
            desired={s:t.copy() for s,t in targets.items()}
            for t in desired.values():
                t[2,3]+=dz
                if mode=='bimanual_side':t[:3,3]+=.002*t[:3,2]
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
    from support_collision import Checker
    checker=Checker(aid)
    yaw=0
    from supportfinger_candidates import candidates
    specs,geometry_counts=candidates(aid)
    print('SUPPORT_GEOMETRY',aid,len(specs),geometry_counts,flush=True)
    # Visit distinct contact regions before changing torso seed.
    for profile in profiles:
      for candidate in specs:
        yaw=candidate['yaw']
        for gap in (.04,.10):
          if time.monotonic()>deadline:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_counts=geometry_counts)
          pos=np.array([.36+candidate['extent_x']/2+gap,0,support_height+.002])
          targets={}
          for side,key in [('l','left'),('r','right')]:
            targets[side]=np.array(candidate[key]['pose']);targets[side][:3,3]+=pos
          seed={**profile,**seedarm,'idx04_body_joint4':0.,'idx05_body_joint5':0.,'idx31_gripper_l_inner_joint1':-.785,'idx71_gripper_r_inner_joint1':-.785}
          p=path(targets,seed,np.eye(4),'bimanual_under',pos,candidate)
          if p:
            p.update(object_yaw_deg=yaw,strategy='actual_finger_branch_under_support',hold_gripper_q=-.785,required_four_finger_contact=True,initial_pose_scope='pregrasp; descent separately audited')
            hit=checker.check(p)
            if hit:
              reject['object_contact_'+hit['link']]+=1
              continue
            out.append(p)
          if len(out)>=max_plans:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_counts=geometry_counts)
    return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_counts=geometry_counts)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--support-height',type=float,default=0.);a=p.parse_args()
    d=plan(a.asset,a.support_height);out=ROOT/'strategy15_supportfinger_plans';out.mkdir(exist_ok=True);(out/(a.asset+'.json')).write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='plans'}),'plans',len(d['plans']),flush=True)
