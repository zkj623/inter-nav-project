from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v4';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v3/box_white.json').read_text());d['plans'][0]['repair_revision']='v4 early forward clearance plus continuous total-load and normalized finger balance controller'
(O/'box_white.json').write_text(json.dumps(d,indent=2))
s=(R/'realpose_box_reinforced_trial.py').read_text();s=s.replace("'gripper' not in name else 10.)","'gripper' not in name else 20.)").replace("else 10. if 'gripper' in n","else 20. if 'gripper' in n")
s=s.replace("'kp':10.","'kp':20.").replace("(1.75 if phase!='hold' else 4.)","(1.75 if phase!='hold' else 1.)")
ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b)
print('WHITE_V4_READY')
