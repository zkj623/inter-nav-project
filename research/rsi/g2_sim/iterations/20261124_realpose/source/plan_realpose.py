from pathlib import Path
import json,copy,argparse,numpy as np,time
from g2_kinematics import Kinematics
from liftall_collision import check,space
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;OLD=R.parent/'g2-act-collect-20261122';pa=argparse.ArgumentParser();pa.add_argument('--asset',required=True);pa.add_argument('--dx',type=float,default=0.);args=pa.parse_args();cfg=json.loads((R/'ground_lift_training_config.json').read_text());oldprep=json.loads((R/'baseline_configs/standby_preparation_config.json').read_text());e=oldprep['execution'][args.asset];doc=json.loads((R/e['plan_dir']/(args.asset+'.json')).read_text());old=doc['plans'][e['candidate']];p=copy.deepcopy(old);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');fixed={n:v for n,v in cfg['common_initial_joint_positions_rad'].items() if 'body_joint' in n or 'head_joint' in n};base=np.array(old['base_pose']);tcp=np.array(old['tcp']);O=R/'realpose_plans';O.mkdir(exist_ok=True);report=dict(asset=args.asset,success=False,scope='fixed realpose body and head; recompute arms preserving original world TCP/object poses',errors=[]);start=time.time()
def solve(q,seed):
    targets=k.fk(q,base);qq={**q,**fixed}
    for side in ['l','r']:
        target=targets[f'arm_{side}_end_link']@tcp;target[0,3]+=args.dx
        arm={n:v for n,v in seed.items() if f'arm_{side}_joint' in n}
        qq,pe,re=k.solve(side,target,{**qq,**arm},base,tcp)
        report['errors'].append(dict(side=side,position_m=pe,rotation_rad=re))
        if pe>.0015 or re>.015:raise ValueError(f'IK_UNREACHABLE {side} {pe} {re}')
    return qq
try:
    p['object_position']=(np.array(old['object_position'])+[args.dx,0,0]).tolist()
    p['realpose_object_shift_m']=[args.dx,0,0]
    report['object_shift_x_m']=args.dx
    for key in ['q_pre','q_grasp','q_close']:p[key]=solve(old[key],old[key])
    for key in ['q_approach','q_lift']:p[key]=[solve(q,q) for q in old.get(key,[])]
    p['q_start']=cfg['common_initial_joint_positions_rad'];p['q_default']=p['q_start'].copy();p['q_prepare']=[]
    p['fixed_body']={**p.get('fixed_body',{}),**fixed};p['profile_id']=cfg['profile_id'];p['collection_initialization']='direct realpose low-home initialization; no standby transition in this test'
    hit=check(args.asset,p)
    if hit:raise ValueError('GRASP_GEOMETRY '+str(hit))
    armnames=[n for n in k.by_name if 'arm_' in n];trajectory=[p['q_start']]+p.get('q_approach',[])+[p['q_pre'],p['q_grasp'],p['q_close']]+p['q_lift'];hits=[]
    for a,b in zip(trajectory,trajectory[1:]):
        count=max(3,int(max(abs(a.get(n,0)-b.get(n,0)) for n in armnames)/.06)+2)
        for u in np.linspace(0,1,count):
            q={n:float((1-u)*a.get(n,0)+u*b.get(n,0)) for n in set(a)|set(b)}
            h=intersections(q)
            if h:raise ValueError('VISIBLE_SELF '+str(h))
    report['success']=True;(O/(args.asset+'.json')).write_text(json.dumps(dict(asset=args.asset,plans=[p]),indent=2))
except Exception as error:report['reason']=str(error)
report['seconds']=time.time()-start;(O/(args.asset+'_planning.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)
raise SystemExit(0 if report['success'] else 2)
