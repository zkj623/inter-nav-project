from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;O=R/'repair_v8';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v7/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v8 centered rim grasp; limit force-relief aperture to-0.025rad;0.8Nm master gripper cap prevents excessive pinch impulse';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_basket_rectangular_trial.py';s=f.read_text();old="        if 'body_joint' in joint['name']:joint['effort']=1e7";new=old+"\n        elif 'gripper' in joint['name'] and not joint['mimic']:joint['effort']=min(joint['effort'],.8)";assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('rate*elapsed,-.12,0.)','rate*elapsed,-.025,0.)').replace('offset[-0.12,0]rad','offset[-0.025,0]rad').replace("'effort_limits':'unchanged'","'effort_limits':'master grippers capped at0.8Nm; arms unchanged'")
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V8_READY')
