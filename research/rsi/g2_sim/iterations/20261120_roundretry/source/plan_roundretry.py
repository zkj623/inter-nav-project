from pathlib import Path
import json,copy,numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
a='bin_round_lidded';original=json.loads((R/'directprep_plans'/f'{a}.json').read_text())['plans'][0];plans[a]=original;base=np.array(original['base_pose']);home=original['q_start'];fkold=k.fk(original['q_close'],base);centerlocal=original['candidate']['contact_centers_local'];objpos=np.array(original['object_position']);variants=[];out=R/'roundretry_plans';out.mkdir(exist_ok=True)
# Sidewall radius is interpolated from original visual mesh rings, below lid.
v=np.load(R/'object_assets/bin_round_lidded.npz')['vertices'];zz=np.unique(np.round(v[:,2],7));rings=[]
for z in zz:
 vv=v[abs(v[:,2]-z)<1e-6]
 if len(vv)>=24 and .12<z<.31:rings.append((z,float(np.max(abs(vv[:,1])))))
for h,vertical_sign in [(h,s) for h in [.25,.28,.30] for s in [1,-1]]:
 objpos[0]=.44;p0=original.copy();p0['object_position']=objpos.tolist();original=p0
 r=float(np.interp(h,[x[0] for x in rings],[x[1] for x in rings]));slope=.083;col1=np.array([0,slope,1.]);col1/=np.linalg.norm(col1);col3=np.array([0,-1.,slope]);col3/=np.linalg.norm(col3);Rleft=np.column_stack([col1*vertical_sign,np.cross(col3,col1)*vertical_sign,col3]);ref=np.diag([1,-1,1]);rots={'l':Rleft,'r':ref@Rleft@ref};p=copy.deepcopy(original);ok=True;solved={}
 for label,clearance in [('q_pre',.035),('q_grasp',.0015),('q_close',.0002)]:
  q=home.copy();q.update(idx31_gripper_l_inner_joint1=-.6,idx71_gripper_r_inner_joint1=-.6)
  for side in ('l','r'):
   c=objpos+np.array([0,(r+clearance)*(1 if side=='l' else -1),h]);t=np.eye(4);t[:3,:3]=rots[side];t[:3,3]=c-rots[side]@np.array(centerlocal[side]);best=None
   rng=np.random.default_rng(17);sn=[n for n in q if f'arm_{side}_joint' in n];seeds=[q,{**q,**{n:x for n,x in original[label].items() if 'arm_' in n}}]+[{**q,**{n:float(rng.uniform(k.by_name[n]['lower']+.15,k.by_name[n]['upper']-.15)) for n in sn}} for _ in range(8)]
   for seed in seeds:
    q2,pe,re=k.solve(side,t,seed,base,np.eye(4));margin=min(min(q2[n]-j['lower'],j['upper']-q2[n]) for n,j in k.by_name.items() if f'arm_{side}_joint' in n)
    score=pe*50+re+sum((q2[n]-home[n])**2 for n in q2 if f'arm_{side}_joint' in n)*.001
    if pe<.002 and re<.02 and margin>.06 and (best is None or score<best[0]):best=(score,q2)
   if best is None:print('IK_DETAIL',label,side,pe,re,margin,flush=True);ok=False;break
   q.update({n:x for n,x in best[1].items() if f'arm_{side}_joint' in n})
  if not ok:break
  p[label]=q;solved[label]=q
 if not ok:print('IK_FAIL',h,vertical_sign,flush=True);continue
 p['q_approach']=[p['q_pre']];p['q_lift']=[];last=p['q_close'].copy()
 for dz in np.linspace(.025,.20,8):
  q=last.copy();f=k.fk(p['q_close'],base)
  for side in ('l','r'):
   t=f[f'arm_{side}_end_link'].copy();t[2,3]+=dz;q,pe,re=k.solve(side,t,q,base,np.eye(4));ok=ok and pe<.002 and re<.02
  p['q_lift'].append(q);last=q
 if not ok:print('LIFT_IK_FAIL',h,vertical_sign,flush=True);continue
 p['candidate']['height_fraction']=h/.401627;p['candidate']['vertical_finger_alignment']=True;p['candidate']['vertical_sign']=vertical_sign;p['candidate']['sidewall_slope']=slope;p['strategy']='four_finger_vertical_sidewall_support';p['hold_gripper_q']=-.6;plans[a]=p
 hit=check(a,p);route=edge(a,home,p['q_pre']);print('CANDIDATE',h,vertical_sign,'hit',hit,'route',route,flush=True)
 if hit or not route:continue
 p['candidate_id']=len(variants);variants.append(p);(out/f'{a}.json').write_text(json.dumps(dict(asset=a,plans=variants),indent=2))
(R/'roundretry_planning.json').write_text(json.dumps(dict(accepted=len(variants),radius_rings=rings),indent=2));print('ROUNDRETRY_PLANNED',len(variants),flush=True)
