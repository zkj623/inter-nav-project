from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;O=R/'repair_v10';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v9/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v10 v9 actual-wall grasp; half-speed lift to reduce contact transients; retain strict surface checks';(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
f=R/'realpose_basket_rectangular_trial.py';s=f.read_text();s=s.replace("seconds=max(seconds,need)*(1.75 if phase!='hold' else 1.)","seconds=max(seconds,need)*(3.5 if phase=='lift' else 1.75 if phase!='hold' else 1.)")
needle='        before_clock=world.current_time';insert='''        if step%12==0:
            pre_object_pose=mat(*obj.get_world_pose())
''';assert s.count(needle)==1;s=s.replace(needle,insert+needle)
needle='                    od/=1000.;fd/=1000.';insert='''                    pre_oi=(pts-pre_object_pose[:3,3])@pre_object_pose[:3,:3]
                    _,pre_od,_=trimesh.proximity.closest_point(omesh_mm,pre_oi*1000.);pre_od/=1000.
''';assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace('object_error_m=od.tolist(),finger_error_m=fd.tolist()','object_error_m=od.tolist(),prestep_object_error_m=pre_od.tolist(),finger_error_m=fd.tolist()')
ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('BASKET_V10_READY')
