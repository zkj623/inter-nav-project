from pathlib import Path
import json,base64,urllib.request,fcntl
import h5py,numpy as np
W=Path(__file__).resolve().parent;rows=json.loads((W/'datasets/pilot4/manifest.json').read_text());rng=np.random.default_rng(7);out=[]
def post(path,request):
 data=json.dumps(request).encode();req=urllib.request.Request('http://127.0.0.1:19029'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)
with (W/'gpu0_simulation.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for run,step in [('act_pilot4_refine',3000),('act_pilot4_robust',2000)]:
  info=post('/load',dict(checkpoint=str(W/f'runs/{run}/step_{step:06d}.pt')));measure=[]
  for asset in ['box_white','bin_lidded']:
   row=next(r for r in rows if r['asset']==asset)
   with h5py.File(row['cache']) as f:
    for t in [0,150,400,700]:
     q=f['qpos'][t];images=base64.b64encode(np.stack([f[n][t] for n in ['head_front','wrist_left','wrist_right']]).tobytes()).decode();request=dict(images=images,qpos=q.tolist(),asset=asset,checkpoint_sha256=info['sha256']);clean=np.array(post('/predict',request)['actions']);perturb=rng.normal(0,.02,14);noisy=q.copy();noisy[:14]+=perturb;request['qpos']=noisy.tolist();changed=np.array(post('/predict',request)['actions']);measure.append(dict(asset=asset,frame=t,absolute_target_sensitivity=float(np.linalg.norm(changed[0,:14]-clean[0,:14])/np.linalg.norm(perturb)),clean_first_action_mae=float(abs(clean[0]-f['action'][t]).mean())))
  out.append(dict(run=run,step=step,mean_sensitivity=float(np.mean([r['absolute_target_sensitivity'] for r in measure])),samples=measure))
(W/'state_sensitivity_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
