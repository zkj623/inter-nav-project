from pathlib import Path
import json,copy,numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from gravity_compensation import GravityCompensation
from liftall_collision import k,robot,space,check,cache
from audit_self_collision import intersections
R=Path(__file__).resolve().parent;g=GravityCompensation();a='stool_wood_four_leg';base=np.eye(4);base[2,3]=.04;rows=json.loads((R/'torquelow_stoolcone_search_results.json').read_text());candidates=sorted([r for r in rows if r['geometry_clear']],key=lambda r:(r['torso_tilt_deg']>40.05,r['torso_tilt_deg']));original=json.loads((R/'directprep_plans'/f'{a}.json').read_text())['plans'][0];tcp=np.array(original['tcp']);oldfk=k.fk(original['q_grasp'],base);oldyaw=Rotation.from_euler('z',original.get('object_yaw_deg',0),degrees=True).as_matrix();local=oldyaw.T@((oldfk['arm_l_end_link']@tcp)[:3,3]-np.array(original['object_position']));arms=[n for n in candidates[0]['q'] if 'arm_' in n];body=[n for n in candidates[0]['q'] if 'body_joint' in n];jointnames=[n for n,j in k.by_name.items() if j['kind']=='revolute' and not j['mimic'] and ('arm_' in n or 'body_joint' in n)];limits={n:48. if 'body_joint' in n else .95*k.by_name[n]['effort'] for n in jointnames};accepted=[]
for idx,row in enumerate(candidates):
 q=row['q'];f=k.fk(q,base);contact=f['arm_l_end_link']@tcp;theta=row['task'][1];psi=np.radians(theta+90);objrot=Rotation.from_euler('z',psi+np.radians(original.get('object_yaw_deg',0))).as_matrix();objpos=contact[:3,3]-objrot@local;objpos[2]=.002;p=copy.deepcopy(original);p.update(mode='single_top',strategy='single_arm_seat_edge_with_counterweight_arm',object_position=objpos.tolist(),object_yaw_deg=float(np.degrees(psi)+original.get('object_yaw_deg',0)),q_start=q.copy(),q_default=q.copy(),q_prepare=[],torso_tilt_limit_deg=45.,required_positive_contacts=2,waist_effort_limit_nm=50.,source_search_row=idx,source_pose=row,static_payload_model='1kg at estimated object horizontal COM, not at grasp point');p['candidate']=dict(original['candidate'],single_arm='l',source_left=original['candidate']['left'],contact_orientation_changed_from_original=True)
 def effort(qq,dz,loaded):
  fk=k.fk(qq,base);tau=g.torques(qq,base)
  if loaded:
   com=objpos+np.array([0,0,.23+dz])
   for n in jointnames:
    if 'arm_l_end_link' not in g.e.desc[n]:continue
    j=k.by_name[n];t=fk[j['parent']]@j['origin'];axis=t[:3,:3]@j['axis'];tau[n]-=float(np.dot(np.cross(com-t[:3,3],[0,0,-9.81]),axis))
  return tau
 def solve(target,seed,dz,loaded):
  # Most waypoints require only left-arm IK. Use the other arm for small
  # counterweight adjustments only if the real object COM exhausts margin.
  q2,pe,re=k.solve('l',target,seed,base,tcp);tau=effort(q2,dz,loaded)
  if pe<.002 and re<.02 and all(abs(tau[n])<=limits[n] for n in jointnames) and not space['collision'](q2) and not intersections(q2):return q2
  leftnames=[n for n in arms if 'arm_l_' in n];rng=np.random.default_rng(20261121+round(dz*1000))
  for attempt in range(12):
   alt=seed.copy()
   for n in leftnames:alt[n]=float(np.clip(seed[n]+rng.normal(0.,.35 if attempt<6 else .7),k.by_name[n]['lower']+.03,k.by_name[n]['upper']-.03))
   qa,pa,ra=k.solve('l',target,alt,base,tcp);ta=effort(qa,dz,loaded)
   if pa<.002 and ra<.02 and all(abs(ta[n])<=limits[n] for n in jointnames) and not space['collision'](qa) and not intersections(qa):return qa
  an=arms;lo=[k.by_name[n]['lower']+.02 for n in an];hi=[k.by_name[n]['upper']-.02 for n in an];x0=np.clip([q2[n] for n in an],lo,hi)
  def unpack(x):return {**seed,**dict(zip(an,x))}
  def eq(x):
   ff=k.fk(unpack(x),base)['arm_l_end_link']@tcp;return np.r_[ff[:3,3]-target[:3,3],Rotation.from_matrix(target[:3,:3].T@ff[:3,:3]).as_rotvec()]
  def ine(x):
   qq=unpack(x);tt=effort(qq,dz,loaded);ff=k.fk(qq,base);gaps=[robot['body_link5'].compound_gap(robot[n],np.linalg.inv(ff['body_link5'])@ff[n])-.002 for n in ['arm_l_link2','arm_l_link3','arm_l_link4','arm_l_link5']];return np.r_[[limits[n]-tt[n] for n in jointnames],[limits[n]+tt[n] for n in jointnames],gaps]
  fit=minimize(lambda x:float(np.sum((x-x0)**2)),x0,bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=ine)],method='SLSQP',options=dict(maxiter=150,ftol=1e-8));qq=unpack(fit.x)
  return qq if max(abs(eq(fit.x)))<.002 and min(ine(fit.x))>-.001 else None
 normal=contact[:3,2];pre=contact.copy();pre[:3,3]-=.07*normal;qp=solve(pre,q,0.,False)
 if qp is None:print('PRE_IK_FAIL',idx,flush=True);continue
 for n in [n for n in qp if 'gripper' in n]:qp[n]=-.785
 qg=solve(contact,qp,0.,False)
 if qg is None:print('GRASP_IK_FAIL',idx,flush=True);continue
 for n in [n for n in qg if 'gripper' in n]:qg[n]=-.785
 qc=solve(contact,qg,0.,True)
 if qc is None:print('CLOSE_IK_FAIL',idx,flush=True);continue
 qc['idx31_gripper_l_inner_joint1']=0.;qc['idx71_gripper_r_inner_joint1']=-.6
 p.update(q_pre=qp,q_grasp=qg,q_close=qc,q_lift=[],q_start=qp.copy(),q_default=qp.copy(),q_approach=[qp.copy()],hold_gripper_q=0.)
 last=qc;good=True;torque_rows=[]
 for dz in np.linspace(.025,.20,8):
  t=contact.copy();t[2,3]+=dz;qn=solve(t,last,float(dz),True)
  if qn is None:good=False;break
  qn['idx31_gripper_l_inner_joint1']=0.;qn['idx71_gripper_r_inner_joint1']=-.6;p['q_lift'].append(qn);last=qn;torque_rows.append(dict(lift_m=float(dz),torques_nm=effort(qn,float(dz),True)))
 if not good:print('LIFT_IK_FAIL',idx,flush=True);continue
 hit=check(a,p);trajectory=[qp,qg,qc]+p['q_lift'];selfhits=[dict(i=i,hits=intersections(qq),native=space['collision'](qq)) for i,qq in enumerate(trajectory)];selfhits=[x for x in selfhits if x['hits'] or x['native']];print('PLAN',idx,'object',hit,'self',selfhits,'waistmax',max(abs(t[n]) for t in [effort(qp,0.,False),effort(qg,0.,False)]+[r['torques_nm'] for r in torque_rows] for n in body),flush=True)
 if hit or selfhits:continue
 p['planned_static_torque_rows']=torque_rows;p['support_height_m']=0.;p['scope']='EXPERIMENTAL strict50Nm single-arm seat pinch; one-time ready-pose initialization; no standing transition; waist free original limits; original material/mass/geometry';accepted.append(p)
 if len(accepted)>=2:break
out=R/'torquelow_stoolsafe_plans';out.mkdir(exist_ok=True);(out/f'{a}.json').write_text(json.dumps(dict(asset=a,plans=accepted),indent=2));print('LOW_STOOL_PLANS',len(accepted),flush=True)
