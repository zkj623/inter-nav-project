"""Physical lift-and-hold acceptance; base contact is diagnostic only."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
PROTOCOL = 'lock10_lift_v1'
def allowed_base_contact(event):
    return Path(event['actor']).name == 'base_link'
def hold_metrics(hold):
    if not hold: return dict(accepted=False)
    tail = [s for s in hold if s['t'] >= hold[-1]['t'] - 3.0 - 1e-6]
    lifts = np.array([s['lift_m'] for s in tail])
    rel = np.array([s['object_relative_tcp'] for s in tail])
    rotations = Rotation.from_quat(np.array([s['object_quaternion'] for s in tail])[:, [1,2,3,0]])
    angular = float(np.degrees((rotations[0].inv()*rotations).magnitude()).max())
    drift = float(np.linalg.norm(rel-rel[0],axis=1).max())
    span = float(hold[-1]['t']-hold[0]['t'])
    variation = float(np.ptp(lifts))
    accepted = span >= 4.9-1e-6 and len(tail)>=29 and min(s['lift_m'] for s in hold)>=.09 and variation<=.02 and drift<=.02 and angular<=10.
    return dict(accepted=bool(accepted),hold_span_s=span,final_window_s=float(tail[-1]['t']-tail[0]['t']),height_variation_m=variation,relative_drift_m=drift,orientation_change_deg=angular)
