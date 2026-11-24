from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v6';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v5/box_white.json').read_text());d['plans'][0]['repair_revision']='v6 v5 support geometry; stronger open-gripper support;30N per-hand target with faster normalized load balance';(O/'box_white.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_box_white_trial.py';s=f.read_text();assert "force_error=20.-total_load" in s
s=s.replace("force_error=20.-total_load","force_error=30.-total_load").replace("if max(loads)>35.:rate=-.0008","if max(loads)>40.:rate=-.0008")
s=s.replace("np.radians(.6)*feedback_dt*imbalance","np.radians(2.)*feedback_dt*imbalance")
s=s.replace("-np.radians(8),np.radians(8)","-np.radians(12),np.radians(12)")
s=s.replace("'gripper' not in name else 20.)","'gripper' not in name else 40.)").replace("else 20. if 'gripper' in n","else 40. if 'gripper' in n").replace("'kp':20.","'kp':40.")
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V6_READY')
