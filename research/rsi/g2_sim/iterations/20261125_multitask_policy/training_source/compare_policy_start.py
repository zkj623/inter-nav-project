from pathlib import Path
import json,h5py,numpy as np,cv2
W=Path(__file__).resolve().parent
rows=json.loads((W/'datasets/pilot4/manifest.json').read_text())
evaluation=json.loads((W/'evaluations/act_pilot4_step003000/summary.json').read_text())
out=[]
for result in evaluation['results']:
 row=next(r for r in rows if r['asset']==result['asset']);trial=Path(result['trial']);expert=Path(row['trial'])
 def episode(p):return next(p.glob('episode*.hdf5'))
 with h5py.File(episode(trial)) as p,h5py.File(episode(expert)) as e:
  q=p['observations/qpos'][:];a=p['action'][:];names=json.loads(p.attrs['joint_names']);delta=q[0]-e['observations/qpos'][0];images=[];imdiff={}
  for name in ['head_front','wrist_left','wrist_right']:
   pi=p['observations/images/'+name][0];ei=e['observations/images/'+name][0];imdiff[name]=float(abs(pi.astype(float)-ei).mean());images.append(cv2.resize(np.concatenate([ei,pi],axis=1),(800,330)))
  cv2.imwrite(str(W/(result['asset']+'_start_comparison.jpg')),cv2.cvtColor(np.concatenate(images),cv2.COLOR_RGB2BGR))
  out.append(dict(asset=result['asset'],frames=len(q),qpos_start_max_difference=float(abs(delta).max()),image_mae=imdiff,first_target_vs_expert_max=float(abs(a[0]-e['action'][0]).max()),last_target_qpos_error={n:float(v) for n,v in zip(names,a[-1]-q[-1]) if abs(v)>.1}))
(W/'start_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
