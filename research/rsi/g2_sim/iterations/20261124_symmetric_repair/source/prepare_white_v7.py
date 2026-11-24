from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;O=R/'repair_v7';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v5/box_white.json').read_text());d['plans'][0]['repair_revision']='v7 v5 support geometry/controller; only double open-gripper stiffness40 to resist collapse';(O/'box_white.json').write_text(json.dumps(d,indent=2))
r=json.loads((R/'realpose_trials/repair_v5/box_white/candidate0_repeat1/result.json').read_text());s=(R/'source_versions'/(r['source_sha256']+'.py')).read_text();assert 'force_error=20.-total_load' in s
s=s.replace("'gripper' not in name else 20.)","'gripper' not in name else 40.)").replace("else 20. if 'gripper' in n","else 40. if 'gripper' in n").replace("'kp':20.","'kp':40.")
ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V7_READY')
