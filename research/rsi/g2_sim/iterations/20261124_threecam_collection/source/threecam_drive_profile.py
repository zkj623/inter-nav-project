"""Recover and verify fixed low-level gains from the exact archived controller."""
from pathlib import Path
import ast,json,hashlib,numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent
def profile_for_trial(T):
 T=Path(T);res=json.loads((T/'result.json').read_text());src=R/'source_versions'/(res['source_sha256']+'.py');assert hashlib.sha256(src.read_bytes()).hexdigest()==res['source_sha256'];tree=ast.parse(src.read_text());k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');names=[n for n,j in k.by_name.items() if j['kind']!='fixed'];env=dict(np=np,k=k,names=names)
 # Execute only the archived fixed effort-cap loop and gain array expressions,
 # never the simulator, planner, controller or any external commands.
 loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='joint' and isinstance(n.iter,ast.Attribute) and isinstance(n.iter.value,ast.Name) and n.iter.value.id=='k' and n.iter.attr=='joints'];assert len(loops)==1
 exec(compile(ast.Module(body=loops,type_ignores=[]),str(src),'exec'),env)
 for key in ['kp','kd']:
  nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==key for x in n.targets)];assert len(nodes)==1;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),env)
 actual={n:dict(kp=float(env['kp'][i]),kd=float(env['kd'][i]),total_effort_cap_nm=float(0. if k.by_name[n]['mimic'] else k.by_name[n]['effort'])) for i,n in enumerate(names)}
 if 'executed_drive_profile' in res:
  for n,d in res['executed_drive_profile'].items():assert actual[n]==d,(n,actual[n],d)
 cfg=json.loads((R/'ground_lift_training_config.json').read_text());policy={n:actual[n] for n in cfg['policy_action_joint_names']};pid=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()[:16]
 out=dict(profile_id=pid,asset=res['asset'],source_sha256=res['source_sha256'],policy_joint_drives=policy,gravity_compensation='feed-forward plus residual feedback within recorded total effort cap',scope='Per-asset fixed low-level profile; use this profile during ACT rollouts. No expert object-pose or force-balancing controller is required for replay.')
 (T/'execution_profile.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 import sys
 p=profile_for_trial(sys.argv[1]);print(json.dumps(dict(asset=p['asset'],profile_id=p['profile_id'],policy_joint_drives=p['policy_joint_drives'])),flush=True)
