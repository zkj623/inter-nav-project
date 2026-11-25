from pathlib import Path
import json,sys,h5py,numpy as np
W=Path(__file__).resolve().parent
run=sys.argv[1];summary=json.loads((W/'evaluations'/run/'summary.json').read_text());rows=json.loads((W/'datasets/pilot4/manifest.json').read_text());records=[]
for result in summary['results']:
 asset=result['asset'];trial=Path(result['trial']);path=trial/'episode.hdf5'
 if not path.exists():path=trial/'episode.partial.hdf5'
 row=next(r for r in rows if r['asset']==asset)
 with h5py.File(row['cache']) as f:expert=f['qpos'][:,:14];phase=f['phase'].asstr()[:]
 with h5py.File(path) as f:
  q=f['observations/qpos'][:];times=f['timestamps'][:];trace=[]
  for i in np.linspace(0,len(q)-1,8,dtype=int):
   errors=np.sqrt(np.mean((expert-q[i,:14])**2,axis=1));j=int(np.argmin(errors))
   trace.append(dict(record_time_s=float(times[i]),policy_elapsed_s=float(times[i]-times[0]),nearest_demo_time_s=j/20,nearest_demo_phase=str(phase[j]),nearest_demo_arm_rmse_deg=float(errors[j]*180/np.pi)))
 contact_path=trial/'nonhand_contacts.json';contacts=json.loads(contact_path.read_text()) if contact_path.exists() else []
 policy=result.get('policy') or {};r={k:result.get(k) for k in ['asset','success','failure','max_lift_m','min_hold_lift_m','max_hold_tilt_deg','trial','video']}
 r.update(trajectory=trace,nonhand_contact_links=sorted(set(x.get('actor','unknown') for x in contacts)),clipped_value_fraction=policy.get('clipped_values',0)/max(1,policy.get('requested_values',0)),frames=len(q));records.append(r)
out=dict(run=run,checkpoint=summary['checkpoint'],completed_trials=len(records),successes=sum(x['success'] for x in records),results=records,interpretation='Nearest demonstration posture is a distance diagnostic, not inferred ground-truth phase. Data are training-position rollouts, not held-out generalization.')
path=W/'pi05_focus_20261125'/(run+'_trajectory_diagnostics.json');path.write_text(json.dumps(out,indent=2));print(json.dumps(out))
