"""Separate grasp-path feasibility from the old standing-to-low descent path."""
from pathlib import Path
import json,subprocess,os
R=Path(__file__).resolve().parent
P=R/'strategy15_lowpose_plans';P.mkdir(exist_ok=True)
for aid in ['chair_dining','chair_wheeled']:
    plans=[]
    for variant in ['guard','chairfar','chair']:
        src=R/f'strategy15_{variant}_plans'/(aid+'.json')
        for i,p in enumerate(json.loads(src.read_text())['plans']):
            p.pop('q_start',None)
            p.update(source_variant=variant,source_candidate=i,initial_pose_scope='low pregrasp; descent not validated')
            plans.append(p)
    (P/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=plans),indent=2))
for name in ['validate_plans','audit_native_self','audit_object_clearance']:
    source=(R/f'strategy15_guard_{name}.py').read_text().replace('strategy15_guard','strategy15_lowpose')
    (R/f'strategy15_lowpose_{name}.py').write_text(source)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for aid in ['chair_dining','chair_wheeled']:
    for name in ['validate_plans','audit_native_self','audit_object_clearance']:
        subprocess.run(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(R/f'strategy15_lowpose_{name}.py'),aid],cwd=R,env=env,check=True)
