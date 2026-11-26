
from pathlib import Path
import json,html,datetime
import numpy as np
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent
R=W.parent/'g2-act-threecam-20261124'
O=W/'pi05_regular240_20261126'
rows=[]
for p in sorted((W/'evaluations').glob('pi05_regular240_*_heldout/summary.json')):
 d=json.loads(p.read_text());step=int(d['run'].split('_')[2])
 item=dict(run=d['run'],steps=step,equivalent_epochs=step*8/238645,completed_objects=d['total'],strict_passes=d['successes'],objects=[])
 probe=W/'pi05_focus_20261125'/(d['run']+'_offline_probe.json')
 if probe.exists():
  ss=json.loads(probe.read_text())['samples']
  item['heldout_arm_first10_mae_rad']=float(np.mean([s['arm_first10_mae_rad'] for s in ss]))
  item['heldout_gripper_first10_mae_native']=float(np.mean([s['gripper_first10_mae_native'] for s in ss]))
  item['probe_states']=len(ss)
 for row in d['results']:
  t=Path(row['trial']);sample=t/'samples.json'
  r={k:row.get(k) for k in ['asset','success','failure','max_lift_m','rejections']}
  r['trial']=str(t)
  r['video']=str(W/'pi05_focus_20261125/reviews'/d['run']/(row['asset']+'.mp4'))
  if sample.exists():
   ss=json.loads(sample.read_text());hold=[s for s in ss if s['phase']=='hold']
   r['hold_span_s']=hold[-1]['t']-hold[0]['t'] if hold else 0
   if hold:
    v=np.load(R/'object_assets'/(row['asset']+'.npz'))['vertices']
    gaps=[]
    for s in hold:
     rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix()
     gaps.append(float((v@rot.T+np.array(s['object_position']))[:,2].min()))
    r['min_hold_lift_m']=min(s['lift_m'] for s in hold)
    r['min_mesh_floor_clearance_m']=min(gaps)
    r['airborne_ten_seconds']=bool(r['hold_span_s']>=9.9-1e-6 and r['min_hold_lift_m']>=.09 and min(gaps)>.005)
    rel=np.array([s['object_relative_tcp'] for s in hold])
    rots=Rotation.from_quat(np.array([s['object_quaternion'] for s in hold])[:,[1,2,3,0]])
    r['hold_relative_motion_m']=float(np.linalg.norm(rel-rel[0],axis=1).max())
    r['hold_rotation_change_deg']=float(np.degrees((rots[0].inv()*rots).magnitude()).max())
  item['objects'].append(r)
 item['airborne_ten_seconds_count']=sum(r.get('airborne_ten_seconds',False) for r in item['objects'])
 rows.append(item)
rows.sort(key=lambda r:r['steps'])
report=dict(updated=datetime.datetime.now().isoformat(),stages=rows,notes=[
 '216 train and 24 heldout episodes; each stage evaluates one heldout placement per object.',
 'Same per-object policy sampling seed and commanded placement; not exact simulator state forks.',
 'Continuous policy for a fixed per-object demonstration-derived duration; assess final10s. Base contact alone is not failure.',
 'Report sustained full-mesh airborne behavior separately from strict acceptance.',
 'This evaluates active holding at the end of the rollout, not chassis transport.',
 'One rollout per object per checkpoint does not estimate a robust success rate.'])
tmp=O/'checkpoint_comparison.tmp.json';tmp.write_text(json.dumps(report,indent=2));tmp.replace(O/'checkpoint_comparison.json')
parts=['<!doctype html><meta charset="utf-8"><title>Regular240 checkpoint comparison</title><style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:30px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Regular240: checkpoint comparison</h1>','<p>'+html.escape(' '.join(report['notes']))+'</p>','<table><tr><th>Steps</th><th>Epochs</th><th>Heldout arm MAE (rad)</th><th>Fully airborne 10s</th><th>Strict pass</th></tr>']
for r in rows:
 parts.append(f"<tr><td>{r['steps']}</td><td>{r['equivalent_epochs']:.2f}</td><td>{r.get('heldout_arm_first10_mae_rad')}</td><td>{r['airborne_ten_seconds_count']}/{r['completed_objects']}</td><td>{r['strict_passes']}/{r['completed_objects']}</td></tr>")
parts.append('</table>')
for r in rows:parts.append('<h2>'+html.escape(r['run'])+'</h2><pre>'+html.escape(json.dumps(r,indent=2))+'</pre>')
(O/'checkpoint_comparison.html').write_text('\n'.join(parts))
print('COMPARISON_READY',len(rows))
