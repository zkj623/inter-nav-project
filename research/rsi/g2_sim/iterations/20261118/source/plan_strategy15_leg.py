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
from strategy15_leg_candidates import make_candidates
import fcl
from audit_self_collision import intersections
ROOT=Path(__file__).resolve().parent
def plan(aid,support_height=0.,max_plans=6,single_only=False):
    started=time.monotonic();deadline=started+600.;env=Envelope();k=env.k;m=mesh(aid);base=transform((0,0,.04));reject=Counter();out=[];attempts=0
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
        body_names=['idx01_body_joint1','idx02_body_joint2','idx03_body_joint3']
        names=(body_names if allow_body else [])+sum((armnames[s] for s in targets),[]);x0=np.array([q.get(n,0.) for n in names])
        lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);x0=np.clip(x0,lo+1e-6,hi-1e-6)
        def residual(x):
            poses=k.fk({**q,**dict(zip(names,x))},base);r=[]
            for s,t in targets.items():
                actual=poses[f'arm_{s}_end_link']@tcp
                r.extend(4*(actual[:3,3]-t[:3,3]));r.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
            body_sum=sum(dict(zip(names,x)).get(n,q.get(n,0.)) for n in body_names)
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
        avoid_plane=pos[0]-.5*m.extents[1 if yaw in (90,270) else 0]-.015
        grasp=solve(targets,seed,tcp,allow_body=True)
        if grasp is None:return None
        # Do not let an orientation solver silently fold the body again.
        payload=[(f'arm_{s}_end_link',1/len(targets),tcp[:3,3]) for s in targets]
        if not env.check(grasp,payload)['ok']:reject['grasp_static_effort']+=1;return None
        pretargets={s:t.copy() for s,t in targets.items()}
        for t in pretargets.values():t[:3,3]-=.09*t[:3,2]
        pre=solve(pretargets,grasp,tcp)
        if pre is None:return None
        closed=grasp.copy()
        if mode=='bimanual_side':
            squeeze={s:t.copy() for s,t in targets.items()}
            for t in squeeze.values():t[:3,3]+=.002*t[:3,2]
            closed=solve(squeeze,grasp,tcp)
            if closed is None:return None
        else:
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
    opened={'idx31_gripper_l_inner_joint1':-.785,'idx71_gripper_r_inner_joint1':-.785}
    fkopen=k.fk(opened);tcp=transform((0,0,.13));oshape=bvh(m);finger_shapes={};finger_local={}
    for side in ('l','r'):
        inv=np.linalg.inv(fkopen[f'arm_{side}_end_link']@tcp)
        for pathfile in (ROOT/'robot_assets').glob(f'gripper_{side}_*.npz'):
            n=pathfile.stem;finger_shapes[n]=bvh(mesh(n));finger_local[n]=inv@fkopen[n]
    for yaw in ((90,0) if aid in ('basket_woven','basket_rectangular','box_white','case_handheld') else (180,270,0,90)):
        rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
        candidates=make_candidates(aid)
        bands={'basket_woven':(.72,1.),'basket_rectangular':(.65,.99),'stool_wood_four_leg':(.88,1.),'stool_rectangular':(.65,1.),'table_side':(.87,1.),'chair_dining':(.12,.35),'chair_wheeled':(.08,.30),'stool_small':(0.,.18),'box_white':(.45,.90),'case_handheld':(.30,.99)}
        low,high=bands.get(aid,(.35,.88));candidates=[c for c in candidates if low <= (np.array(c['pose'])[2,3]-m.bounds[0,2])/m.extents[2] <= high][:250]
        expanded=[]
        for c in candidates:
            if c.get('axis')==2:
                expanded.append(c);continue
            old=np.array(c['pose']);closing=old[:3,0];perp=np.cross(np.array([0.,0.,1.]),closing)
            if np.dot(perp,old[:3,3])>0:perp=-perp
            for approach in (perp, -perp, (perp+np.array([0.,0.,-1.]))/np.sqrt(2.), np.array([0.,0.,-1.])):
                t=old.copy();t[:3,:3]=np.column_stack([closing,np.cross(approach,closing),approach])
                expanded.append({**c,'pose':t.tolist(),'approach_object':approach.tolist()})
        candidates=expanded
        left=[c for c in candidates if (rot@np.array(c['pose'])[:3,3])[1]>.04]
        right=[c for c in candidates if (rot@np.array(c['pose'])[:3,3])[1]<-.04]
        def individually_clear(c,side):
            for offset in (.09,.04,0.):
                target=np.array(c['pose']);target[:3,3]-=offset*target[:3,2]
                for n,shape in finger_shapes.items():
                    if not n.startswith(f'gripper_{side}_'):continue
                    pose=target@finger_local[n];res=fcl.CollisionResult();fcl.collide(fcl.CollisionObject(shape,fcl.Transform(pose[:3,:3],pose[:3,3])),fcl.CollisionObject(oshape),fcl.CollisionRequest(),res)
                    if res.is_collision:return False
            return True
        left=[c for c in left if individually_clear(c,'l')]
        right=[c for c in right if individually_clear(c,'r')]
        print('INDIVIDUAL_CLEAR',aid,yaw,len(left),len(right),flush=True)
        left.sort(key=lambda c:float((rot@np.array(c['pose'])[:3,3])[0]))
        right.sort(key=lambda c:float((rot@np.array(c['pose'])[:3,3])[0]))
        for lc in left[:20]:
            for rc in sorted(right,key=lambda c:abs(np.array(c['pose'])[2,3]-np.array(lc['pose'])[2,3])+float((rot@np.array(c['pose'])[:3,3])[0]))[:2]:
                for gap in (.14,.20,.26):
                    pos=np.array([.36+m.extents[1 if yaw in (90,270) else 0]/2+gap,0,support_height+.002])
                    targets={}
                    for side,c in [('l',lc),('r',rc)]:
                        t=np.array(c['pose']);t[:3,:3]=rot@t[:3,:3];t[:3,3]=rot@t[:3,3]+pos;targets[side]=t
                    clear=True
                    for side,c in [('l',lc),('r',rc)]:
                        for offset in (.09,.04,0.):
                            target=np.array(c['pose']);target[:3,3]-=offset*target[:3,2]
                            for n,shape in finger_shapes.items():
                                if not n.startswith(f'gripper_{side}_'):continue
                                pose=target@finger_local[n];res=fcl.CollisionResult();fcl.collide(fcl.CollisionObject(shape,fcl.Transform(pose[:3,:3],pose[:3,3])),fcl.CollisionObject(oshape),fcl.CollisionRequest(),res)
                                if res.is_collision:clear=False;break
                            if not clear:break
                        if not clear:break
                    if not clear:reject['open_finger_geometry_clearance']+=1;continue
                    for profile in profiles:
                        if time.monotonic()>deadline:break
                        seed={**profile,**seedarm,'idx31_gripper_l_inner_joint1':-.785,'idx71_gripper_r_inner_joint1':-.785}
                        result=path(targets,seed,transform((0,0,.13)),'bimanual_top',pos,dict(left=lc,right=rc,gap=gap))
                        if result:
                            result['object_yaw_deg']=float(yaw);result['strategy']='geometry_feature_side_approach';result['hold_gripper_q']=0.;out.append(result)
                        if len(out)>=max_plans:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
    return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--support-height',type=float,default=0.);a=p.parse_args()
    d=plan(a.asset,a.support_height);out=ROOT/'strategy15_leg_plans';out.mkdir(exist_ok=True);(out/(a.asset+'.json')).write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='plans'}),'plans',len(d['plans']),flush=True)
