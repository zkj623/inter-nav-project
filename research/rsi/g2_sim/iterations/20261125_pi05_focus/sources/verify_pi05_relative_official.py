import sys,json,copy
from pathlib import Path
import numpy as np,h5py
w=Path(__file__).resolve().parent;sys.path.insert(0,str(w/'vendor/openpi/src'))
from openpi.transforms import DeltaActions,AbsoluteActions,make_bool_mask
mask=make_bool_mask(14,-2);rows=json.loads((w/'datasets/pilot4/manifest.json').read_text());error=0.;count=0
for row in rows:
 with h5py.File(row['cache'],'r') as f:
  n=len(f['action'])
  for t in [0,n//2,n-1]:
   q=f['qpos'][t];actions=f['action'][np.minimum(np.arange(t,t+50),n-1)] if t+50<=n else np.stack([f['action'][min(i,n-1)] for i in range(t,t+50)])
   expected=actions.copy();expected[:,:14]-=q[:14]
   official=DeltaActions(mask)({'state':q.copy(),'actions':actions.copy()})
   assert np.array_equal(official['actions'],expected)
   assert np.array_equal(expected[:,14:],actions[:,14:])
   restored=AbsoluteActions(mask)(official)['actions'];error=max(error,float(np.max(np.abs(restored-actions))));count+=1
assert error<1e-6
r=dict(passed=True,official_commit='215abfb217dbac7d5f1273282331b9b1866c0479',samples=count,episodes=len(rows),arms='first 14 relative to current measured qpos for entire chunk',grippers='last 2 absolute',official_transform_matches_exactly=True,reconstruction_max_error_rad=error)
(w/'pi05_focus_20261125/official_delta_equivalence_test.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
