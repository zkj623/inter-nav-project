from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';O=W/'pi05_focus_20261125';sys.path.insert(0,str(R))
from terminal_lift_rules import hold_metrics
run='pi05_focus_terminal_relative_129000';targets=json.loads((R/'terminal_lift_targets.json').read_text())['nominal_lift_m'];report=dict(protocol='terminal_lift_v1',checkpoint=129000,policy_continues=True,rows=[],note='One rollout per object; demonstration-height band selects a 10-second observation window, not proof of a completed policy phase. Relative motion uses left TCP; continued intentional movement can affect static stability metrics.')
for asset,target in targets.items():
 ps=list((R/'policy_trials'/run).rglob(asset+'/candidate0_repeat1/result.json'))
 if not ps:continue
 p=ps[0];d=json.loads(p.read_text());ss=json.loads(p.with_name('samples.json').read_text());h=[s for s in ss if s['phase']=='hold'];floor=[]
 if h:
  mesh=np.load(R/'object_assets'/(asset+'.npz'))['vertices']
  for s in h:
   rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();floor.append(float((mesh@rot.T+np.array(s['object_position']))[:,2].min()))
 span=h[-1]['t']-h[0]['t'] if h else 0.
 ap=p.with_name('stable_verification.json');audit=json.loads(ap.read_text()) if ap.exists() else {}
 cp=p.with_name('stable_threecam_acceptance.json');cam=json.loads(cp.read_text()) if cp.exists() else {}
 events=json.loads(p.with_name('nonhand_contacts.json').read_text())
 row=dict(asset=asset,demo_target_m=target,trigger_m=target-.005,reached_demo_band=bool(h),max_lift_m=max((s['lift_m'] for s in ss),default=0),window_span_s=span,min_window_lift_m=min((s['lift_m'] for s in h),default=None),final_lift_m=h[-1]['lift_m'] if h else None,min_visual_floor_clearance_m=min(floor) if floor else None,airborne_ten_seconds=bool(h and span>=9.9-1e-6 and min(s['lift_m'] for s in h)>=.09 and min(floor)>.005),metrics=hold_metrics(h),harness_success=d['success'],verified_success=bool(audit.get('verified_success') and cam.get('accepted')),failure=d.get('failure'),audit_rejections=audit.get('rejections'),nonbase_contact_links=sorted({Path(e['actor']).name for e in events if Path(e['actor']).name!='base_link' and e['impulse']>.001}),base_contact_events=sum(Path(e['actor']).name=='base_link' for e in events),continued_policy_calls=(d.get('policy_continuation') or {}).get('inference_calls_after_trigger'),trial=str(p.parent),video=str(O/'reviews'/run/(asset+'.mp4')))
 if h:
  rel=np.array([s['object_relative_tcp'] for s in h]);rots=Rotation.from_quat(np.array([s['object_quaternion'] for s in h])[:,[1,2,3,0]])
  row['whole_window_drift_m']=float(np.linalg.norm(rel-rel[0],axis=1).max());row['whole_window_rotation_change_deg']=float(np.degrees((rots[0].inv()*rots).magnitude()).max())
 report['rows'].append(row)
report['complete']=len(report['rows'])==4;report['airborne_ten_seconds_count']=sum(r['airborne_ten_seconds'] for r in report['rows']);report['verified_count']=sum(r['verified_success'] for r in report['rows'])
(O/'terminal_129000_physical_outcomes.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
