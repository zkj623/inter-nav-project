from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'compact_soft';O.mkdir(exist_ok=True);a='stool_wood_four_leg';d=json.loads((R/'compact_paths'/(a+'.json')).read_text());d['plans'][0]['posture_revision']+='; damped soft10/4 gripper with2Nm master cap to avoid closing contact impulse';(O/(a+'.json')).write_text(json.dumps(d,indent=2))
f=R/f'realpose_{a}_trial.py';s=f.read_text()
for old,new in [("'gripper' not in name else 20.)","'gripper' not in name else 10.)"),("else 20. if 'gripper' in n","else 10. if 'gripper' in n"),("'gripper' not in name else 1.)","'gripper' not in name else 4.)"),("else 1. if 'gripper' in n","else 4. if 'gripper' in n")]:assert old in s,old;s=s.replace(old,new)
needle="        if 'body_joint' in joint['name']:joint['effort']=1e7";assert needle in s;s=s.replace(needle,needle+"\n        elif 'gripper' in joint['name'] and not joint['mimic']:joint['effort']=min(joint['effort'],2.)")
s=s.replace("    result['drive_assumption']=","    result['repair_gripper_gains']={'kp':10.,'kd':4.,'effort_limits':'master grippers2Nm cap; arms unchanged'}\n    result['drive_assumption']=")
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('READY_COMPACT_SOFT')
