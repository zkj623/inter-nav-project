from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v13';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v12/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v13 deeper actual-wall pinch; stiffness40 damping4 to maintain wall compression under load;2Nm cap retained';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_basket_rectangular_trial.py';s=f.read_text()
for old,new in [("'kp':10.0","'kp':40.0"),("'kd':2.0","'kd':4.0"),("'gripper' not in name else 10.0)","'gripper' not in name else 40.0)"),("else 10.0 if 'gripper' in n","else 40.0 if 'gripper' in n"),("'gripper' not in name else 2.0)","'gripper' not in name else 4.0)"),("else 2.0 if 'gripper' in n","else 4.0 if 'gripper' in n")]:
 assert old in s,old;s=s.replace(old,new)
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V13_READY')
