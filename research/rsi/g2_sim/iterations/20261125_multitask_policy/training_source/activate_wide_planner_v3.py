from pathlib import Path
import hashlib,json,datetime
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';p=R/'plan_wide_threecam.py';old=p.read_bytes();versions=R/'source_versions';versions.mkdir(exist_ok=True)
(versions/(hashlib.sha256(old).hexdigest()+'.py')).write_bytes(old)
s=old.decode().replace('from wide_posture_planner import','from wide_posture_planner_v3 import').replace("planner_version='20deg_local_search_v2'","planner_version='adaptive20_35_50deg_local_search_v3'")
p.write_text(s);new=p.read_bytes();(versions/(hashlib.sha256(new).hexdigest()+'.py')).write_bytes(new)
(W/'wide_planner_revision_v3.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),old_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(new).hexdigest(),change='Adaptive planning search radius only; original joint bounds, 1.5mm/0.015rad IK tolerance,15deg mirror,geometry and all physical acceptance checks retained'),indent=2))
print('WIDE_PLANNER_V3_ACTIVE_FUTURE_ATTEMPTS')
