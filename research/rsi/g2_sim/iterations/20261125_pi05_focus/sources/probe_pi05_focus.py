from pathlib import Path
import argparse,json,base64,urllib.request
import h5py,numpy as np
W=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,default=19026);p.add_argument('--dataset',default='pilot4');p.add_argument('--output',required=True);p.add_argument('--split',default='train');a=p.parse_args()
def post(path,data):
 request=urllib.request.Request(f'http://127.0.0.1:{a.port}'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=600) as r:return json.load(r)
info=post('/load',dict(checkpoint=a.checkpoint));rows=json.loads((W/'datasets'/a.dataset/'manifest.json').read_text());rows=[r for r in rows if r['split']==a.split];results=[]
for asset in dict.fromkeys(r['asset'] for r in rows):
 row=next(r for r in rows if r['asset']==asset)
 with h5py.File(row['cache']) as f:
  n=len(f['action'])
  for t in np.unique(np.linspace(0,n-1,12,dtype=int)):
   images=np.stack([f[name][t] for name in ['head_front','wrist_left','wrist_right']]);response=post('/predict',dict(images=base64.b64encode(images.tobytes()).decode(),qpos=f['qpos'][t].tolist(),asset=asset,checkpoint_sha256=info['sha256']));pred=np.array(response['actions']);count=min(len(pred),n-t);error=abs(pred[:count]-f['action'][t:t+count]);results.append(dict(asset=asset,frame=int(t),phase=f['phase'].asstr()[t],first_mae_rad=float(error[0].mean()),first10_mae_rad=float(error[:10].mean()),chunk_mae_rad=float(error.mean())));print('PI05_PROBE',results[-1],flush=True)
out=dict(checkpoint=info,per_asset={asset:{key:float(np.mean([r[key] for r in results if r['asset']==asset])) for key in ['first_mae_rad','first10_mae_rad','chunk_mae_rad']} for asset in dict.fromkeys(r['asset'] for r in results)},samples=results)
Path(a.output).write_text(json.dumps(out,indent=2));print('PI05_PROBE_COMPLETE',json.dumps(out['per_asset']),flush=True)
