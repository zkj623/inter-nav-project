from pathlib import Path
import argparse,json,copy,numpy as np,time,os
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,cache,check,space
from audit_self_collision import intersections
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--asset',required=True);parser.add_argument('--index',type=int,required=True);parser.add_argument('--full-preparation',action='store_true');parser.add_argument('--dx',type=float);parser.add_argument('--dy',type=float);parser.add_argument('--yaw',type=float);args=parser.parse_args();cfg=json.loads((R/'standby_preparation_config.json').read_text());e=cfg['execution'][args.asset];old=json.loads((R/e['plan_dir']/(args.asset+'.json')).read_text())['plans'][e['candidate']];p=copy.deepcopy(old);rng=np.random.default_rng(202611220000+args.index*100+list(cfg['execution']).index(args.asset));scale=.005 if args.index<=4 else .01 if args.index<=12 else .02;dx=float(rng.uniform(-scale,scale)) if args.dx is None else args.dx;dy=float(rng.uniform(-scale,scale)) if args.dy is None else args.dy;yaw=float(rng.uniform(-3 if args.index<=12 else -5,3 if args.index<=12 else 5)) if args.yaw is None else args.yaw
out=R/'collection_plans'/f'placement_{args.index:04d}';out.mkdir(parents=True,exist_ok=True);file=out/(args.asset+'.json');start=time.monotonic();report=dict(asset=args.asset,index=args.index,dx_m=dx,dy_m=dy,dyaw_deg=yaw,success=False);base=np.array(old['base_pose']);rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix();center=np.array(old['object_position']);newcenter=center+np.array([dx,dy,0.]);tcp=np.array(old['tcp']);arms=[n for n,j in k.by_name.items() if 'arm_' in n and j['kind']=='revolute'];moved=[]
def solve(q,seed):
 fk=k.fk(q,base);qq=q.copy()
 for side in ('l','r'):
  target=fk[f'arm_{side}_end_link']@tcp;target[:3,3]=newcenter+rot@(target[:3,3]-center);target[:3,:3]=rot@target[:3,:3];qq,pe,re=k.solve(side,target,{**qq,**{n:seed[n] for n in arms if n in seed and f'arm_{side}_joint' in n}},base,tcp)
  # Preserve the other arm when solving the second side.
  for n in qq:
   if f'arm_{side}_joint' not in n and n in q and 'arm_' not in n:qq[n]=q[n]
  if pe>.0015 or re>.015:raise ValueError('IK_UNREACHABLE '+side+' '+str((pe,re)))
 return qq
try:
 p['object_position']=newcenter.tolist();p['object_yaw_deg']=old.get('object_yaw_deg',0)+yaw
 for key in ('q_pre','q_grasp','q_close'):p[key]=solve(old[key],old[key])
 for key in ('q_approach','q_lift'):p[key]=[solve(q,q) for q in old.get(key,[])]
 # Public work pose and standby never move with the object.
 p['q_start']=old['q_start'].copy()
 if not args.full_preparation:p['q_default']=p['q_start'].copy();p['q_prepare']=[]
 p['collection_initialization']='full_standby_preparation' if args.full_preparation else 'one-time common low-home initialization before physics; no standing transition in this episode'
 p['collection_placement']=dict(report,seed=202611220000+args.index*100+list(cfg['execution']).index(args.asset),original_object_position_m=old['object_position'],original_object_yaw_deg=old.get('object_yaw_deg',0),new_object_position_m=p['object_position'],new_object_yaw_deg=p['object_yaw_deg'],group_id=f'{args.asset}_placement_{args.index:04d}')
 hit=check(args.asset,p)
 if hit:raise ValueError('GRASP_GEOMETRY '+str(hit))
 obj=cache[args.asset];objrot=Rotation.from_euler('z',p['object_yaw_deg'],degrees=True).as_matrix()
 def clear(q,dz=0.,allowfingers=False):
  hit=space['collision'](q)
  if hit:return str(hit)
  h=intersections(q)
  if h:return 'visible_self '+str(h)
  fk=k.fk(q,base)
  for n,g in robot.items():
   if allowfingers and n.startswith('gripper_'):continue
   pose=fk[n].copy();pose[:3,3]=objrot.T@(pose[:3,3]-newcenter-np.array([0.,0.,dz]));pose[:3,:3]=objrot.T@pose[:3,:3];c=pose[:3,:3]@g.bounds.mean(0)+pose[:3,3];ext=abs(pose[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
   if np.any(c+ext<obj.bounds[0]) or np.any(c-ext>obj.bounds[1]):continue
   if obj.compound_gap(g,pose)<-.001:return 'object '+n
  return None
 trajectory=[(p['q_start'],0.,False)]+[(q,0.,False) for q in p.get('q_approach',[])]+[(p['q_pre'],0.,False),(p['q_grasp'],0.,False),(p['q_close'],0.,True)]+[(q,.025*(i+1),True) for i,q in enumerate(p['q_lift'])]
 samples=0
 for (q0,z0,f0),(q1,z1,f1) in zip(trajectory,trajectory[1:]):
  steps=max(3,int(max(abs(q0.get(n,0)-q1.get(n,0)) for n in arms)/.06)+2)
  for u in np.linspace(0,1,steps):
   qq={n:(1-u)*q0.get(n,0)+u*q1.get(n,0) for n in set(q0)|set(q1)};h=clear(qq,(1-u)*z0+u*z1,f0 or f1);samples+=1
   if h:raise ValueError('PATH_GEOMETRY '+h)
 report.update(success=True,checked_samples=samples,plan=str(file.relative_to(R)),seconds=time.monotonic()-start);p['collection_placement']['success']=True;(file).write_text(json.dumps(dict(asset=args.asset,plans=[p]),indent=2))
except Exception as exc:report.update(reason=str(exc),seconds=time.monotonic()-start)
(out/(args.asset+'_planning.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
raise SystemExit(0 if report['success'] else 2)

