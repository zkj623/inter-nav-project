from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from liftall_collision import k,robot,CookedGeometry,ConvexPart
R=Path(__file__).resolve().parent;records=[]
for file in sorted(R.glob('liftall_*_trials/**/result.json')):
 d=json.loads(file.read_text());sf=file.with_name('samples.json');samples=json.loads(sf.read_text()) if sf.exists() else []
 row=dict(asset=d['asset'],result=str(file.relative_to(R)),harness_success=d['success'],failure=d.get('failure'),max_lift_m=d.get('max_lift_m'),min_hold_lift_m=d.get('min_hold_lift_m'),max_hold_tilt_deg=d.get('max_hold_tilt_deg'),four_contact_fraction=d.get('hold_four_finger_contact_fraction'),source_sha256=d['source_sha256'])
 row['verified_success']=False
 if samples:
  row['frozen_error_rad']=max(s.get('frozen_joint_max_error_rad',0.) for s in samples)
  row['root_displacement_m']=max(float(np.linalg.norm(np.array(s['base_position'])-[0,0,.04])) for s in samples)
  row['min_body_margin_deg']=float(min(np.degrees(min(s['joint_positions'][n]-j['lower'],j['upper']-s['joint_positions'][n])) for s in samples for n,j in k.by_name.items() if 'body_joint' in n))
 if d['success']:
  aid=d['asset'];pr=R/'object_assets'/(aid+'_cache_provenance.json');path=json.loads(pr.read_text())['path'] if pr.exists() else R/'object_assets'/(aid+'_cooked.json')
  if aid in ['stool_rectangular','stool_wood_four_leg','table_side','basket_rectangular']:path=R/'strategy15_surface_assets'/(aid+'_cooked.json')
  dd=json.loads(Path(path).read_text());obj=CookedGeometry([ConvexPart(h['vertices_object_m'],str(i)) for i,h in enumerate(dd['hulls'])]);hits=[];worst=0.;floor=[]
  vertices=np.load(R/'object_assets'/(aid+'.npz'))['vertices']
  for s in samples:
   base=np.eye(4);base[:3,3]=s['base_position'];fk=k.fk(s['joint_positions'],base)
   rot=Rotation.from_quat(np.array(s['object_quaternion'])[[1,2,3,0]]).as_matrix();pos=np.array(s['object_position'])
   if s['phase']=='hold':floor.append(float((vertices@rot.T+pos)[:,2].min()))
   for name,g in robot.items():
    if name.startswith('gripper_'):continue
    pose=fk[name].copy();pose[:3,3]=rot.T@(pose[:3,3]-pos);pose[:3,:3]=rot.T@pose[:3,:3]
    c=pose[:3,:3]@g.bounds.mean(0)+pose[:3,3];e=np.abs(pose[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
    if np.any(c+e<obj.bounds[0]) or np.any(c-e>obj.bounds[1]):continue
    gap=obj.compound_gap(g,pose);worst=min(worst,gap)
    if gap<-.001:hits.append(dict(t=s['t'],phase=s['phase'],link=name,gap_m=gap))
  hold=[s for s in samples if s['phase']=='hold']
  row.update(min_nonhand_gap_m=worst,geometry_hits=hits,min_hold_floor_clearance_m=min(floor),hold_samples=len(hold))
  nc=json.loads(file.with_name('nonhand_contacts.json').read_text());row['max_nonhand_impulse_ns']=max([x['impulse'] for x in nc]+[0.])
  row['verified_success']=bool(not hits and min(floor)>.005 and len(hold)>=29 and min(s['lift_m'] for s in hold)>=.15 and max(s['tilt_deg'] for s in hold)<=20 and row['frozen_error_rad']<.005 and row['root_displacement_m']<.001 and row['min_body_margin_deg']>=9.9 and row['four_contact_fraction']>=.9 and row['max_nonhand_impulse_ns']<=.001)
 for name in ['samples.json','review.mp4']:
  f=file.with_name(name)
  if f.exists():row[name]=dict(server_path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 records.append(row);print(json.dumps(row),flush=True)
(R/'liftall_verified_results.json').write_text(json.dumps(records,indent=2))
