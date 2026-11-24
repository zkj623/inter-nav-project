from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;O=R/'repair_v6';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v5/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v6 yaw0 rim grasp with bilateral jaw centering and four-branch preload BEFORE lift';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_basket_rectangular_trial.py';s=f.read_text();old="    pinch_aperture={'l':0.,'r':0.};pinch_force_time={'l':None,'r':None}"
new=old+'''
    pinch_shift={'l':0.,'r':0.};pinch_axes={}
    grasp_fk=k.fk(plan['q_grasp'],np.array(plan['base_pose']))
    for side in ('l','r'):
        centers=[(grasp_fk[f'gripper_{side}_{finger}_link4']@np.r_[fmeshes[f'gripper_{side}_{finger}_link4'].vertices.mean(0),1.])[:3] for finger in ('inner','outer')]
        axis=grasp_fk[f'arm_{side}_end_link'][:3,:3].T@(centers[0]-centers[1]);pinch_axes[side]=axis/max(1e-9,np.linalg.norm(axis))
'''
assert s.count(old)==1;s=s.replace(old,new)
needle="                total=sum(loads)"
insert="""                if min(loads)<3. and max(loads)>4.:
                    pinch_shift[side]=float(np.clip(pinch_shift[side]+.0005*elapsed*np.sign(loads[0]-loads[1]),-.008,.008))
                target[:3,3]+=pinch_shift[side]*(desired[f'arm_{side}_end_link'][:3,:3]@pinch_axes[side])
"""
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace("achieved=all(sum(v for n,v in forces.items() if f'gripper_{side}_' in n)>.75 for side in ('l','r'))","achieved=four_contact(samples[-1],1.)")
s=s.replace("if plan.get('mode')=='bimanual_spread' and stable<3:","if stable<3:")
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V6_READY')
