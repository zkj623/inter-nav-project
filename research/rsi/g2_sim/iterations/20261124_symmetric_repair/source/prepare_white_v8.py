from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v8';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v7/box_white.json').read_text());d['plans'][0]['repair_revision']='v8 v7 grasp; require four branches>2N for8 settle updates before3s hold';(O/'box_white.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_box_white_trial.py';s=f.read_text();assert 'ready=four_contact(recent,1.)' in s;s=s.replace('ready=four_contact(recent,1.)','ready=four_contact(recent,2.)').replace('if settled_updates>=4:break','if settled_updates>=8:break').replace("if settled_updates<4:raise TrialRejected('NO_SETTLED_FOUR_FINGER_GRASP')","if settled_updates<8:raise TrialRejected('NO_SETTLED_FOUR_FINGER_GRASP')")
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V8_READY')
