from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';O=W/'pi05_focus_20261125'
report=dict(protocol='lock10_lift_v1',height_hold_definition='Secondary diagnostic only: 10 cm trigger, >=4.9s sampled hold, >=9cm reference lift throughout, >5mm visual floor clearance throughout, zero post-lock inference. Does not waive or relabel original independent audit.',single_rollout_per_object=True,stages=[])
for step in [86000,129000,140000]:
 run=f'pi05_focus_paired_relative_{step}_lock10';p=W/'evaluations'/run/'summary.json'
 if not p.exists():continue
 d=json.loads(p.read_text());stage=dict(checkpoint=step,rows=[])
 for row in d['results']:
  t=Path(row['trial']);p=t/'result.json'
  if not p.exists():stage['rows'].append(dict(asset=row['asset'],infrastructure_failure=True));continue
  res=json.loads(p.read_text());samples=json.loads((t/'samples.json').read_text());hold=[s for s in samples if s['phase']=='hold'];clearance=[]
  if hold:
   mesh=np.load(R/'object_assets'/(row['asset']+'.npz'))['vertices']
   for s in hold:
    rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();clearance.append(float((mesh@rot.T+np.array(s['object_position']))[:,2].min()))
  lock=res.get('policy_lock') or {};span=hold[-1]['t']-hold[0]['t'] if hold else 0.
  height_ok=bool(hold and span>=4.9-1e-6 and min(s['lift_m'] for s in hold)>=.09 and min(clearance)>.005 and lock.get('inference_calls_after_lock')==0)
  contacts=json.loads((t/'nonhand_contacts.json').read_text());links=sorted({Path(x['actor']).name for x in contacts})
  item=dict(asset=row['asset'],verified_success=row['success'],height_hold_diagnostic=height_ok,max_lift_m=res.get('max_lift_m'),min_hold_lift_m=res.get('min_hold_lift_m'),hold_span_s=span,min_visual_floor_clearance_m=min(clearance) if clearance else None,lock_metrics=res.get('lock10_metrics'),post_lock_inference_calls=lock.get('inference_calls_after_lock'),failure=row.get('failure'),rejections=row.get('rejections'),nonhand_contact_links=links,base_contact_count=sum(Path(x['actor']).name=='base_link' for x in contacts),max_contact_surface_error_m=res.get('max_contact_surface_error_m'),video=str(O/'reviews'/run/(row['asset']+'.mp4')))
  stage['rows'].append(item)
 stage['verified_successes']=sum(x.get('verified_success',False) for x in stage['rows']);stage['height_hold_count']=sum(x.get('height_hold_diagnostic',False) for x in stage['rows']);report['stages'].append(stage)
report['complete']=len(report['stages'])==3 and all(len(s['rows'])==4 for s in report['stages'])
(O/'paired_lock10_physical_outcomes.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
