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
def plan(aid,support_height=0.,max_plans=6,single_only=False):
    started=time.monotonic();deadline=started+360.;env=Envelope();k=env.k;m=mesh(aid);base=transform((0,0,.04));reject=Counter();out=[];attempts=0
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
    def solve(targets,q,tcp):
        nonlocal attempts
        if time.monotonic()>deadline:reject['search_time_budget']+=1;return None
        fk=k.fk(q,base)
        for s,t in targets.items():
            if np.linalg.norm(t[:3,3]-fk[f'arm_{s}_link1'][:3,3])>reach[s]+np.linalg.norm(tcp[:3,3]):reject['outside_arm_reach_bound']+=1;return None
        names=sum((armnames[s] for s in targets),[]);x0=np.array([q.get(n,0.) for n in names])
        lo=np.array([k.by_name[n]['lower'] for n in names]);hi=np.array([k.by_name[n]['upper'] for n in names]);x0=np.clip(x0,lo+1e-6,hi-1e-6)
        def residual(x):
            poses=k.fk({**q,**dict(zip(names,x))},base);r=[]
            for s,t in targets.items():
                actual=poses[f'arm_{s}_end_link']@tcp
                r.extend(4*(actual[:3,3]-t[:3,3]));r.extend(Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).as_rotvec())
            return np.r_[r,.002*(x-x0)]
        for restart in range(5):
            if restart:x0=np.random.default_rng(restart).uniform(np.maximum(lo,-1.5),np.minimum(hi,1.5))
            attempts+=1;fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=160,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            r=residual(fit.x)
            if not any(np.linalg.norm(r[i:i+3])>.016 or np.linalg.norm(r[i+3:i+6])>.04 for i in range(0,6*len(targets),6)):
                return {**q,**dict(zip(names,fit.x))}
        reject['arm_ik_with_fixed_torso']+=1;return None
    def path(targets,seed,tcp,mode,pos,candidate):
        grasp=solve(targets,seed,tcp)
        if grasp is None:return None
        # Do not let an orientation solver silently fold the body again.
        payload=[(f'arm_{s}_end_link',1/len(targets),tcp[:3,3]) for s in targets]
        if not env.check(grasp,payload)['ok']:reject['grasp_static_effort']+=1;return None
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
            fixed_body={n:v for n,v in seed.items() if 'body_joint' in n},support_height_m=support_height,torso_envelope={'max_tilt_deg':20,'max_body_joint_deg':180,'static_effort_fraction':.8},
            envelope_checks=checks,scope='fixed torso; arms-only IK/lift; initial stance search; no navigation or object attachment')
    # Four separated finger contacts. Open fingers stay open during lift.
    # Each side is checked against BOTH ray intersections at the actual finger spread.
    openq=-.6;tip_depth=.1532;halfspread=.050
    ranges={'bin_lidded':(.48,.58,.40),'bin_pedal':(.78,.65,.50),'bin_round_lidded':(.80,.65,.50),'bin_round_pedal':(.55,.70,.40),
      'box_white':(.65,.50,.78),'box_reinforced':(.65,.50,.78),'case_handheld':(.55,.70,.40),
      'stool_small':(.10,.20,.35),'stool_rectangular':(.70,.80,.55),'stool_wood_four_leg':(.94,.86,.70),
      'table_side':(.95,.85,.70),'chair_dining':(.30,.40,.50),'chair_wheeled':(.15,.18,.45),
      'basket_woven':(.65,.75,.50),'basket_rectangular':(.70,.55,.80)}
    specs=[]
    for yaw in (0,90):
      rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix();mm=m.copy();mm.apply_transform(transform(rpy=(0,0,np.radians(yaw))))
      for frac in ranges[aid]:
       z=mm.bounds[0,2]+mm.extents[2]*frac
       for roll in (np.pi/2,0.,-np.pi/2,np.pi):
        x=np.array([np.cos(roll),0,np.sin(roll)])
        rays=[];normals=[]
        for shift in (-halfspread,halfspread):
          origin=np.array([0.,mm.bounds[0,1]-.1,z])+shift*x
          hits,_,faces=mm.ray.intersects_location([origin],[[0,1,0]],multiple_hits=True)
          if len(hits)<2:break
          order=np.argsort(hits[:,1]);rays.append((hits[order[0]],hits[order[-1]]));normals.append((mm.face_normals[faces[order[0]]],mm.face_normals[faces[order[-1]]]))
        if len(rays)!=2:reject['missing_two_finger_surface_intersections']+=1;continue
        if any(abs(n[1])<.75 for pair in normals for n in pair):reject['surface_normal_incompatible_with_side_support']+=1;continue
        if any(abs(rays[0][i][1]-rays[1][i][1])>.003 for i in (0,1)):reject['surface_curvature_prevents_four_contact']+=1;continue
        lo=(rays[0][0]+rays[1][0])/2;hi=(rays[0][1]+rays[1][1])/2
        for distance in (.02,.08):
          pos=np.array([.36+mm.extents[0]/2+distance,0,support_height+.002]);targets={}
          for side,point,normal in [('l',hi,np.array([0.,-1,0])),('r',lo,np.array([0.,1,0]))]:
            t=np.eye(4);t[:3,:3]=np.column_stack([x,np.cross(normal,x),normal]);t[:3,3]=point+pos;targets[side]=t
          specs.append((targets,pos,dict(height_fraction=frac,stance_clearance_m=distance,tool_roll_rad=roll,object_yaw_deg=yaw,contact_points_object_rotated=[p.tolist() for pair in rays for p in pair])))
    # Round-robin over object regions instead of exhausting every body pose on one bad target.
    for profile_index in range(len(profiles)):
      for targets,pos,candidate in specs:
        if time.monotonic()>deadline:break
        seed={**profiles[profile_index],**seedarm,'idx31_gripper_l_inner_joint1':openq,'idx71_gripper_r_inner_joint1':openq}
        p=path(targets,seed,transform((-.0045,0,tip_depth)),'bimanual_spread',pos,candidate)
        if p:
          p.update(object_yaw_deg=candidate['object_yaw_deg'],hold_gripper_q=openq,strategy='four_finger_spread_side_support',required_four_finger_contact=True)
          out.append(p)
        if len(out)>=max_plans:return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
      if time.monotonic()>deadline:break

    return dict(asset=aid,plans=out,rejections=dict(reject),ik_calls=attempts,support_height_m=support_height)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--support-height',type=float,default=0.);a=p.parse_args()
    d=plan(a.asset,a.support_height);out=ROOT/'strategy15_spread_plans';out.mkdir(exist_ok=True);(out/(a.asset+'.json')).write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='plans'}),'plans',len(d['plans']),flush=True)
