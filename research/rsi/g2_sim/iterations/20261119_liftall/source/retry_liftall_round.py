from pathlib import Path
import json, os, subprocess, sys, time
import numpy as np
from g2_kinematics import Kinematics
R=Path(__file__).resolve().parent
k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
doc=json.loads((R/'liftall_validated_plans/bin_round_lidded.json').read_text())
for plan in doc['plans']:
    fk=k.fk(plan['q_close'],np.array(plan['base_pose']))
    centers={}
    for side in ('l','r'):
        inv=np.linalg.inv(fk[f'arm_{side}_end_link'])
        normal=np.array([0.,-1. if side=='l' else 1.,0.])
        patches=[]
        for finger in ('inner','outer'):
            name=f'gripper_{side}_{finger}_link4'
            verts=np.load(R/'robot_assets'/(name+'.npz'))['vertices']
            t=fk[name];world=verts@t[:3,:3].T+t[:3,3]
            depth=world@normal
            patch=world[depth>depth.max()-.0002].mean(0)
            patches.append((inv@np.r_[patch,1.])[:3])
        centers[side]=np.mean(patches,axis=0).tolist()
    plan['candidate']['contact_centers_local']=centers
    plan['candidate']['contact_center_provenance']='Mean of distal visual surface patches facing inward at q_close; 0.2mm surface band; wrist local coordinates. Joint targets unchanged.'
out=R/'liftall_metadata_plans';out.mkdir(exist_ok=True)
(out/'bin_round_lidded.json').write_text(json.dumps(doc,indent=2))
print('METADATA_READY',centers,flush=True)
while True:
    events=json.loads((R/'liftall_batch_events.json').read_text())
    if any(e.get('status')=='batch_complete' for e in events):break
    time.sleep(10)
with open(R/'liftall_round_retry.log','w') as log:
    subprocess.run([sys.executable,str(R/'liftall_general_trial.py'),'--asset','bin_round_lidded','--plan-dir','liftall_metadata_plans','--gpu','2'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
