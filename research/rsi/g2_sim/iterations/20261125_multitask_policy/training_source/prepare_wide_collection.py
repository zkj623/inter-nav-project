from pathlib import Path
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124')
s=(R/'plan_threecam.py').read_text()
s=s.replace('import json,numpy as np,copy,argparse,time','import json,numpy as np,copy,argparse,time\nfrom scipy.spatial.transform import Rotation')
s=s.replace('202611240000','202611250000')
start=s.index('scale=.005');end=s.index('\ndef solve(q):')
s=s[:start]+'''scale=.03 if args.index<=24 else .02
delta=np.r_[rng.uniform(-scale,scale,2),0.]
while np.linalg.norm(delta[:2])<.012:delta=np.r_[rng.uniform(-scale,scale,2),0.]
yaw=float(rng.uniform(-5,5)) if args.index%2==0 else 0.
rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix();center=np.array(old['object_position'])
folder=f'wideplacements/p{args.index:04d}';O=R/folder;O.mkdir(parents=True,exist_ok=True)
report=dict(asset=a,index=args.index,seed=seed,dx_m=float(delta[0]),dy_m=float(delta[1]),dyaw_deg=yaw,scale_m=scale,group_id=f'{a}_wide_p{args.index:04d}',success=False)
'''+s[end:]
s=s.replace("target[:3,3]+=delta;x,pe,re=solve_posture(k,side,target,q,B,tcp)","target[:3,3]=center+rz@(target[:3,3]-center)+delta;target[:3,:3]=rz@target[:3,:3];x,pe,re=solve_posture(k,side,target,q,B,tcp)")
s=s.replace("p['object_position']=(np.array(old['object_position'])+delta).tolist();","p['object_position']=(np.array(old['object_position'])+delta).tolist();p['object_yaw_deg']=old.get('object_yaw_deg',0.)+yaw;")
s=s.replace("'new_object_yaw_deg':old.get('object_yaw_deg',0)","'new_object_yaw_deg':p['object_yaw_deg']")
(R/'plan_wide_threecam.py').write_text(s)
s=(R/'run_threecam_collection.py').read_text().replace("default=20)","default=10)").replace('default=120)','default=60)').replace('default=[2,2,2,5,6,6]','default=[5]')
s=s.replace("R/'collection.lock'","R/'wide_collection.lock'").replace("R/'collection_state'","R/'wide_collection_state'").replace("R/'collection_logs'","R/'wide_collection_logs'")
s=s.replace("folder=f'placements/p{i:04d}';repeat=2 if a=='box_white' and i in [1,2] else 1","folder=f'wideplacements/p{i:04d}';repeat=1")
s=s.replace("R/'plan_threecam.py'","R/'plan_wide_threecam.py'").replace("<.001 for r in accepted","<.002 for r in accepted").replace('PLACEMENT_NEAR_DUPLICATE_UNDER_1MM','PLACEMENT_NEAR_DUPLICATE_UNDER_2MM')
s=s.replace('collection_progress.tmp.json','wide_collection_progress.tmp.json').replace('collection_progress.json','wide_collection_progress.json')
(R/'run_wide_collection.py').write_text(s)
print('WIDE_COLLECTION_READY: 1.2-4.2cm radial offsets, optional +/-5deg yaw; original gates unchanged')
