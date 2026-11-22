from pathlib import Path
import json,time,numpy as np
from gravity_compensation import GravityCompensation as Original
from fast_gravity_compensation import GravityCompensation as Fast
R=Path(__file__).resolve().parent;original=Original();fast=Fast();cfg=json.loads((R/'ground_lift_training_config.json').read_text());rng=np.random.default_rng(20261122);cases=[cfg['common_initial_joint_positions_rad'],cfg['default_joint_positions_rad']]
for _ in range(100):cases.append({n:float(rng.uniform(j['lower']+.001,j['upper']-.001)) for n,j in original.k.by_name.items() if j['kind']=='revolute' and not j['mimic']})
error=0.
for q in cases:
 a=original.torques(q);b=fast.torques(q);assert a.keys()==b.keys();error=max(error,max(abs(a[n]-b[n]) for n in a));assert abs(original.potential(q)-fast.potential(q))<1e-8
assert error<1e-9,error
for name,model in [('original',original),('vectorized',fast)]:
 start=time.perf_counter()
 for q in cases:model.torques(q)
 print(name,'seconds',time.perf_counter()-start,flush=True)
print('GRAVITY_EQUIVALENCE_OK',len(cases),'max_abs_nm',error,'source_sha256',fast.implementation_sha256,flush=True)
