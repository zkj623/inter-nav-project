from pathlib import Path
import json,hashlib,ast,copy,numpy as np
R=Path(__file__).resolve().parent;O=R/'natural_v1';O.mkdir(exist_ok=True)
cfg=json.loads((R/'ground_lift_training_config.json').read_text());sign=[-1,1,-1,1,-1,1,-1];nl=[f'idx{21+i}_arm_l_joint{i+1}' for i in range(7)];nr=[f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)];M=np.diag([1.,-1.,1.])
for a,reference in [('basket_rectangular','r'),('basket_woven','r'),('stool_wood_four_leg','l'),('table_side','l')]:
 old=json.loads((Path(cfg['validated_simulation_baselines'][a])/'result.json').read_text());p=copy.deepcopy(old['plan'])
 def mirror(q):
  q=q.copy()
  for ln,rn,s in zip(nl,nr,sign):
   if reference=='r':q[ln]=s*q[rn]
   else:q[rn]=s*q[ln]
  return q
 for key in ['q_pre','q_grasp','q_close']:
  if key in p:p[key]=mirror(p[key])
 p['q_lift']=[mirror(q) for q in p['q_lift']];p['q_approach']=[p['q_pre'].copy()]
 ref='right' if reference=='r' else 'left';other='left' if reference=='r' else 'right';p['candidate'][other]=copy.deepcopy(p['candidate'][ref]);pose=np.array(p['candidate'][ref]['pose'])
 # Metadata in object coordinates: convert reflection through world y=0.
 from scipy.spatial.transform import Rotation
 A=Rotation.from_euler('z',p.get('object_yaw_deg',0),degrees=True).as_matrix();F=A.T@M@A;pose[:3,3]=F@pose[:3,3];pose[:3,:3]=F@pose[:3,:3]@M;p['candidate'][other]['pose']=pose.tolist()
 p['posture_revision']='natural_v1 symmetric nominal joint path; runtime IK anchored to nominal path within10deg to prevent redundant-joint branch drift';p['posture_reference_side']=reference
 (O/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[p]),indent=2))
 f=R/f'realpose_{a}_trial.py';s=f.read_text();needle='solution,pe,re=k.solve(side,target,actual,current_base,tcp)';assert s.count(needle)==1;s=s.replace(needle,'solution,pe,re=solve_posture(k,side,target,nominal,current_base,tcp)')
 s=s.replace('from isaacsim import SimulationApp',"from posture_anchor import solve_posture\nresult['posture_anchor_sha256']=hashlib.sha256((ROOT/'posture_anchor.py').read_bytes()).hexdigest()\nfrom isaacsim import SimulationApp")
 ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b)
 print('READY',a,reference,flush=True)
