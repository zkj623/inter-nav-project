from pathlib import Path
import json,numpy as np,sys
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';O=W/'pi05_focus_20261125'
sys.path.insert(0,str(R))
from lock10_lift_rules import hold_metrics
runs={'fixed_targets':'pi05_focus_paired_relative_129000_lock10','continued_policy':'pi05_focus_continued_relative_129000_continue10'}
report=dict(checkpoint=129000,conditions=runs,rows=[],note='Matched seed and initial conditions; inspect pre-trigger divergence before attributing differences solely to hold control. Continued ascent may exceed height-variation check without being a slip. Relative movement is measured against the left TCP and can include intentional arm adjustment.')
for asset in ['box_white','bin_lidded','bin_pedal','basket_rectangular']:
 records={};traces={}
 for label,run in runs.items():
  files=list((R/'policy_trials'/run).rglob(asset+'/candidate0_repeat1/result.json'))
  if not files:continue
  p=files[0];d=json.loads(p.read_text());ss=json.loads(p.with_name('samples.json').read_text());h=[x for x in ss if x['phase']=='hold'];traces[label]=ss;floor=[]
  if h:
   mesh=np.load(R/'object_assets'/(asset+'.npz'))['vertices']
   for x in h:
    rot=Rotation.from_quat(np.array(x['object_quaternion'])[[1,2,3,0]]).as_matrix();floor.append(float((mesh@rot.T+np.array(x['object_position']))[:,2].min()))
  audit=p.with_name('stable_verification.json');audit=json.loads(audit.read_text()) if audit.exists() else {}
  records[label]=dict(harness_success=d['success'],verified_success=audit.get('verified_success'),failure=d.get('failure'),audit_rejections=audit.get('rejections'),max_lift_m=d.get('max_lift_m'),min_hold_lift_m=min(x['lift_m'] for x in h) if h else None,final_lift_m=h[-1]['lift_m'] if h else None,hold_start_s=h[0]['t'] if h else None,hold_span_s=h[-1]['t']-h[0]['t'] if h else 0,min_visual_floor_clearance_m=min(floor) if floor else None,metrics=hold_metrics(h),control=d.get('policy_continuation') if label=='continued_policy' else d.get('policy_lock'),video=str(O/'reviews'/run/(asset+'.mp4')))
 row=dict(asset=asset,conditions=records)
 if len(records)==2:
  starts=[x['hold_start_s'] for x in records.values() if x['hold_start_s'] is not None]
  if starts:
   cutoff=min(starts)-.1;a={round(s['t'],6):s for s in traces['fixed_targets'] if s['t']<cutoff};pairs=[(a[round(s['t'],6)],s) for s in traces['continued_policy'] if round(s['t'],6) in a and s['t']<cutoff]
   row['pretrigger_comparison']=dict(samples=len(pairs),until_s=cutoff,max_object_position_difference_m=max((float(np.linalg.norm(np.array(x['object_position'])-y['object_position'])) for x,y in pairs),default=None),max_joint_difference_rad=max((max(abs(v-y['joint_positions'][n]) for n,v in x['joint_positions'].items()) for x,y in pairs),default=None))
 report['rows'].append(row)
report['complete']=all(len(x['conditions'])==2 for x in report['rows'])
(O/'continue10_vs_fixed129000.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
