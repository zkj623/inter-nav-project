from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v11';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v10/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v11 actual-wall centered pinch with slow lift;2Nm gripper cap provides load support while remaining below original10Nm limit';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_basket_rectangular_trial.py';s=f.read_text();assert "min(joint['effort'],.8)" in s;s=s.replace("min(joint['effort'],.8)","min(joint['effort'],2.)").replace('capped at0.8Nm','capped at2Nm')
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V11_READY')
