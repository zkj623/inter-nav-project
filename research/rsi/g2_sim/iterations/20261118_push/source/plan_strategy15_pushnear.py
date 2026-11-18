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
def plan(aid,support_height=0.,max_plans=3,single_only=False):
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
    from chair_core_guidance import Guidance
    guide=Guidance(aid)
    seedarm={'idx22_arm_l_joint2':-.6,'idx62_arm_r_joint2':-.6,'idx24_arm_l_joint4':-1.2,'idx64_arm_r_joint4':-1.2}
    def solve(targets,q,tcp,allow_body=False):
        nonlocal attempts
        if time.monotonic()>deadline:reject['search_time_budget']+=1;return None
        fk=k.fk(q,base)
        for s,t in targets.items():
            if not allow_body and np.linalg.norm(t[:3,3]-fk[f'arm_{s}_link1'][:3,3])>reach[s]+np.linalg.norm(tcp[:3,3]):reject['outside_arm_reach_bound']+=1;return None
        body_names=['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3','idx04_body_joint4','idx05_body_joint5']
        names=(body_names if allow_body else [])+sum((armnames[s] for s in ('l','r')),[]);x0=np.array([q.get(n,0.) for n in names])
        lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);x0=np.clip(x0,lo+1e-6,hi-1e-6)
        if allow_body:
            lo[3:5]=[-.30,-1.4];hi[3:5]=[.30,1.4];x0=np.clip(x0,lo+1e-6,hi-1e-6)
        def residual(x):
            poses=k.fk({**q,**dict(zip(names,x))},base);r=[]
            for s,t in targets.items():
                actual=poses[f'arm_{s}_end_link']@tcp
                r.extend(4*(actual[:3,3]-t[:3,3]));r.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
            body_sum=float(np.arccos(np.clip((poses['body_link5'][:3,:3]@env.upright.T)[2,2],-1,1)))
            clearances=[8*max(0.,np.max(corners@poses[bn][0,:3]+poses[bn][0,3])-avoid_plane) for bn,corners in bodycorners.items()]
            return np.r_[r,.002*(x-x0),10*max(0.,abs(body_sum)-np.radians(18.)),clearances,guide.residual(poses)]
        for restart in range(3):
            if restart:
                x0=np.random.default_rng(restart).uniform(np.maximum(lo,-1.5),np.minimum(hi,1.5))
                if allow_body:x0[:len(body_names)]=np.clip([q.get(n,0.) for n in body_names],lo[:len(body_names)]+1e-6,hi[:len(body_names)]-1e-6)
            attempts+=1;fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=160,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            r=residual(fit.x)
            if not any(np.linalg.norm(r[i:i+3])>.016 or np.linalg.norm(r[i+3:i+6])>.04 for i in range(0,6*len(targets),6)):
                return {**q,**dict(zip(names,fit.x))}
        reject['arm_ik_with_fixed_torso']+=1;return None
    from pushnear_candidates import candidates
    from push_collision import Checker
    checker=Checker(aid);specs,geometry_rejections=candidates(aid)
    print('PUSH_GEOMETRY',aid,len(specs),geometry_rejections,flush=True)
    deadline=time.monotonic()+300.
    deep=[dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),np.radians(v))) for v in [(-60,140,-70),(-60,150,-80),(-60,160,-90)]]
    profiles=deep+profiles
    for profile_pass in range(len(profiles)):
      for ci,c in enumerate(specs):
       for gap,lateral in ((.06,0.),(.12,0.),(0.,.12)):
        if time.monotonic()>deadline:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_rejections=geometry_rejections)
        pos=np.array([.36+c['extent_x']/2+gap,lateral,.002]);yaw=c['yaw']
        targets={side:np.array(t) for side,t in c['targets'].items()}
        for t in targets.values():t[:3,3]+=pos
        seed={**profiles[(ci+profile_pass)%len(profiles)],**seedarm,'idx04_body_joint4':0.,'idx05_body_joint5':0.,'idx31_gripper_l_inner_joint1':0.,'idx71_gripper_r_inner_joint1':0.}
        high={side:t.copy() for side,t in targets.items()}
        for t in high.values():t[0,3]+=.15
        guide.set_pose(pos+[.15,0,0],yaw);end=solve(high,seed,np.eye(4),allow_body=True)
        if end is None:reject['push_end_ik']+=1;continue
        pretargets={side:t.copy() for side,t in targets.items()}
        for t in pretargets.values():t[0,3]-=.04
        guide.set_pose(pos,yaw);pre=solve(pretargets,end,np.eye(4))
        if pre is None:reject['pre_ik']+=1;continue
        contact=solve(targets,pre,np.eye(4))
        if contact is None:reject['contact_ik']+=1;continue
        pushes=[];last=contact;offsets=np.linspace(.015,.15,10);failed=False
        for dx in offsets:
         desired={side:t.copy() for side,t in targets.items()}
         for t in desired.values():t[0,3]+=dx
         guide.set_pose(pos+[dx,0,0],yaw);last=solve(desired,last,np.eye(4))
         if last is None:reject['push_path_ik']+=1;failed=True;break
         pushes.append(last)
        if failed:continue
        release_targets={side:t.copy() for side,t in high.items()}
        for t in release_targets.values():t[0,3]-=.04
        release=solve(release_targets,last,np.eye(4))
        if release is None:reject['release_ik']+=1;continue
        ns=set(pre)|set(contact);approach=[{n:(1-u)*pre.get(n,0)+u*contact.get(n,0) for n in ns} for u in np.linspace(0,1,7)]
        states=approach+pushes+[release]
        if not all(env.check(q)['ok'] for q in states):reject['envelope']+=1;continue
        if any(intersections(q) for q in states):reject['visible_self']+=1;continue
        p=dict(asset=aid,mode=c['mode'],candidate=c,object_position=pos.tolist(),object_yaw_deg=yaw,base_pose=base.tolist(),tcp=np.eye(4).tolist(),q_pre=pre,q_grasp=contact,q_close=contact,q_push=pushes,q_lift=pushes,push_offsets_m=offsets.tolist(),q_release=release,support_height_m=0.,hold_gripper_q=0.,scope='prepose initialized; stationary mobile base; physical rigid-chair sliding; no wheel rolling joints; no attachment')
        hit=checker.check(p)
        if hit:reject['object_'+hit['link']]+=1;continue
        out.append(p);print('PUSH_PLAN',aid,len(out),c['contact_height'],yaw,c['mode'],flush=True)
        if len(out)>=max_plans:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_rejections=geometry_rejections)
    return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,geometry_rejections=geometry_rejections)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--asset',required=True);a=p.parse_args();d=plan(a.asset)
 out=ROOT/'strategy15_pushnear_plans';out.mkdir(exist_ok=True);(out/(a.asset+'.json')).write_text(json.dumps(d,indent=2));print({k:v for k,v in d.items() if k!='plans'},'plans',len(d['plans']),flush=True)
