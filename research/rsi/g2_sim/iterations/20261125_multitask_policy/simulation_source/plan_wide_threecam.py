from pathlib import Path
import json,numpy as np,copy,argparse,time
from scipy.spatial.transform import Rotation
from wide_posture_planner_v3 import solve_posture
R=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--asset',required=True);pa.add_argument('--index',type=int,required=True);args=pa.parse_args();start=time.monotonic()
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'geometry_library','exec'))
cfg=json.loads((R/'standby_preparation_config.json').read_text());a=args.asset;e=cfg['execution'][a];old=json.loads((R/e['plan_dir']/(a+'.json')).read_text())['plans'][e['candidate']];p=copy.deepcopy(old);B=np.array(p['base_pose']);tcp=np.array(p['tcp']);seed=202611250000+args.index*100+list(cfg['execution']).index(a);rng=np.random.default_rng(seed)
scale=.03 if args.index<=24 else .02
delta=np.r_[rng.uniform(-scale,scale,2),0.]
while np.linalg.norm(delta[:2])<.012:delta=np.r_[rng.uniform(-scale,scale,2),0.]
yaw=float(rng.uniform(-5,5)) if args.index%2==0 else 0.
rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix();center=np.array(old['object_position'])
folder=f'wideplacements/p{args.index:04d}';O=R/folder;O.mkdir(parents=True,exist_ok=True)
report=dict(asset=a,index=args.index,seed=seed,dx_m=float(delta[0]),dy_m=float(delta[1]),dyaw_deg=yaw,scale_m=scale,group_id=f'{a}_wide_p{args.index:04d}',planner_version='adaptive20_35_50deg_local_search_v3',success=False)

def solve(q):
 qq=q.copy();fk=k.fk(q,B)
 for side in ['l','r']:
  target=fk[f'arm_{side}_end_link']@tcp;target[:3,3]=center+rz@(target[:3,3]-center)+delta;target[:3,:3]=rz@target[:3,:3];x,pe,re=solve_posture(k,side,target,q,B,tcp)
  if pe>.0015 or re>.015:raise ValueError('IK_UNREACHABLE '+str((side,pe,re)))
  qq.update({n:v for n,v in x.items() if f'arm_{side}_joint' in n})

 mirror=max(abs(np.degrees(qq[f'idx{61+i}_arm_r_joint{i+1}']-sg*qq[f'idx{21+i}_arm_l_joint{i+1}'])) for i,sg in enumerate([-1,1,-1,1,-1,1,-1]))
 if mirror>15:raise ValueError('PLANNED_POSTURE_MIRROR_EXCEEDS15DEG')
 return qq
try:
 for key,value in old.items():
  if not key.startswith('q_') or key in ['q_start','q_default','q_prepare']:continue
  if isinstance(value,dict) and any('arm_' in n for n in value):p[key]=solve(value)
  elif isinstance(value,list) and value and isinstance(value[0],dict):p[key]=[solve(q) for q in value]
 p['object_position']=(np.array(old['object_position'])+delta).tolist();p['object_yaw_deg']=old.get('object_yaw_deg',0.)+yaw;p['q_default']=p['q_start'].copy();p['q_prepare']=[];p['collection_initialization']='common low home before physics; body/base fixed; no policy preparation motion';p['collection_placement']={**report,'new_object_position_m':p['object_position'],'original_object_position_m':old['object_position'],'original_object_yaw_deg':old.get('object_yaw_deg',0),'new_object_yaw_deg':p['object_yaw_deg']};plans[a]=p
 cp=dict(p)
 if 'q_contact_approach' in p:cp['q_pre']=p['q_grasp']
 hit=check(a,cp)
 if hit:raise ValueError('GRASP_LIFT_GEOMETRY '+str(hit))
 for key,route in [('approach',[p['q_start']]+p.get('q_approach',[])+[p['q_pre']]),('contact',[p['q_pre']]+p.get('q_contact_approach',[p['q_grasp']]))]:
  if not all(edge(a,x,y) for x,y in zip(route,route[1:])):raise ValueError('PATH_GEOMETRY '+key)
 report.update(success=True,plan_dir=folder);p['collection_placement']['success']=True;(O/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[p]),indent=2))
except Exception as ex:report['failure']=str(ex)
report['seconds']=time.monotonic()-start;(O/(a+'_planning.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
raise SystemExit(0 if report['success'] else 2)
