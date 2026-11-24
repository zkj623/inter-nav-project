"""Load the requested RSI standby pose and measure its enabled collision hull."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def load_standby(path, kinematics):
    path = Path(path)
    config = json.loads(path.read_text())
    q = {name: float(value) for name, value in config['standby_joint_positions_rad'].items()}
    required = {j['name'] for j in kinematics.joints if j['kind'] != 'fixed' and not j['mimic']}
    assert set(q) == required, {'missing': sorted(required-set(q)), 'unknown': sorted(set(q)-required)}
    for name, value in q.items():
        joint = kinematics.by_name[name]
        assert np.isfinite(value), name
        if joint['kind'] == 'revolute':
            assert joint['lower']-1e-7 <= value <= joint['upper']+1e-7, name
    source = {'field': 'standby_joint_positions_rad', 'configured_joints': len(q),
              'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'joint_values_modified': False}
    origin_path = Path(__file__).with_name('standby_pose_origin.json')
    if origin_path.exists():
        origin = json.loads(origin_path.read_text())
        if origin['config_sha256'] == source['config_sha256']:
            source.update(origin)
    return q, source


def collision_footprint(stage, start_xy, yaw, pose_source):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], useExtentsHint=False)
    points = []
    count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        bbox = cache.ComputeWorldBound(prim)
        bounds = bbox.GetRange()
        if bounds.IsEmpty():
            continue
        lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        matrix = np.array(bbox.GetMatrix())
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        world = np.c_[corners, np.ones(8)] @ matrix
        delta = world[:, :2]-np.asarray(start_xy)
        c, s = np.cos(yaw), np.sin(yaw)
        points.append(delta @ np.array([[c, -s], [s, c]]))
        count += 1
    assert points, 'No enabled collision geometry found'
    points = np.concatenate(points)
    hull = ConvexHull(points)
    return {'vertices': points[hull.vertices].tolist(), 'margin_m': .02,
            'source': 'Enabled native collision-part bounds in RSI standby pose; empty bounds excluded',
            'collision_parts_checked': count, 'pose_source': pose_source,
            'bounds_xy_m': [points.min(0).tolist(), points.max(0).tolist()]}
