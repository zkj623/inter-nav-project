"""Audit observation causality independently of manipulation success."""
from pathlib import Path
import argparse,json
import h5py,numpy as np
from g2_force_features import causal_force
p=argparse.ArgumentParser();p.add_argument('--trial',required=True);a=p.parse_args();T=Path(a.trial)
samples=json.loads((T/'samples.json').read_text());episode=T/'episode.hdf5'
if not episode.exists():episode=T/'episode.partial.hdf5'
with h5py.File(episode) as f:
 ts=f['timestamps'][:];force=f['observations/finger_forces'][:];stamp=f['force_timestamps'][:]
 assert len(ts)>0 and len(ts)==len(force)==len(stamp)
 reference=[causal_force(samples,float(t)) for t in ts];expected=np.stack([v[0] for v in reference]);expected_ts=np.array([v[1] for v in reference])
 assert np.allclose(force,expected,rtol=0,atol=1e-9)
 assert np.allclose(stamp,expected_ts,rtol=0,atol=1e-9)
 assert (ts>=stamp).all()
 audit=dict(passed=True,frames=len(ts),max_force_error_n=float(abs(force-expected).max()),max_age_s=float((ts-stamp).max()),causal=True,semantics='Latest three available 10Hz simulator distal finger-object measurements; no object pose or future force input')
(T/'force_input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
