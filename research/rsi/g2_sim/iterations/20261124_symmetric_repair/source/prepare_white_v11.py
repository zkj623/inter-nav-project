from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;O=R/'repair_v11';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v7/box_white.json').read_text());d['plans'][0]['repair_revision']='v11 v7 stable contact geometry; open-gripper stiffness80 damping8 to reduce load-dependent pad deflection; unchanged motor effort limits';(O/'box_white.json').write_text(json.dumps(d,indent=2))
r=json.loads((R/'realpose_trials/repair_v7/box_white/candidate0_repeat1/result.json').read_text());s=(R/'source_versions'/(r['source_sha256']+'.py')).read_text()
s=s.replace("'gripper' not in name else 40.)","'gripper' not in name else 80.)").replace("else 40. if 'gripper' in n","else 80. if 'gripper' in n").replace("'kp':40.","'kp':80.")
s=s.replace("'gripper' not in name else 4.)","'gripper' not in name else 8.)").replace("else 4. if 'gripper' in n","else 8. if 'gripper' in n").replace("'kd':4.","'kd':8.")
ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V11_READY')
