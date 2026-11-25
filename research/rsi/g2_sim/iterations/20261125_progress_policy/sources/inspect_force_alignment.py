from pathlib import Path
import json,h5py,numpy as np
W=Path(__file__).resolve().parent
rows=json.loads((W/'datasets/pilot4_one_each/manifest.json').read_text())
for row in rows:
 p=Path(row['trial']);samples=json.loads((p/'samples.json').read_text())
 with h5py.File(p/'episode.hdf5') as f,h5py.File(row['cache']) as cache:
  ts=f['timestamps'][:];st=np.array([s['physics_time_s'] for s in samples]);idx=np.searchsorted(st,ts,side='right')-1
  assert (idx>=0).all()
  print(json.dumps(dict(asset=row['asset'],frames=len(ts),cache_frames=len(cache['qpos']),q_error=float(abs(f['observations/qpos'][:]-cache['qpos'][:]).max()),max_age=float((ts-st[idx]).max()),min_age=float((ts-st[idx]).min()),sample_keys=list(samples[0]),cache_keys=list(cache),first_time=ts[0],sample_time=st[idx[0]])))
