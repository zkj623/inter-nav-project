from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v9';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v7/box_white.json').read_text());d['plans'][0]['repair_revision']='v9 v7 geometry and stiff open grippers; retain settled targets during hold to eliminate load-feedback chatter';(O/'box_white.json').write_text(json.dumps(d,indent=2))
r=json.loads((R/'realpose_trials/repair_v7/box_white/candidate0_repeat1/result.json').read_text());s=(R/'source_versions'/(r['source_sha256']+'.py')).read_text()
old="""                hold_start=step
                while step-hold_start<360:
                    corrected=retarget(q,last)
                    segment(last,corrected,min(.1,(360-(step-hold_start))/120.),'hold');last=corrected"""
assert s.count(old)==1;s=s.replace(old,"                segment(last,last,3.,'hold')");ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V9_READY')
