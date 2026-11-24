from pathlib import Path
import ast,hashlib,json
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124');W=Path(__file__).resolve().parent
# Version the recorder rather than changing bytes referenced by running trials.
s=(R/'threecam_recorder.py').read_text()
needle='   for _ in range(3):world.render()'
replacement='''   for _ in range(3):world.render()
   if self.count==0:
    # Sensor startup can yield one black left-wrist frame despite valid timestamps.
    # Warm up on the same physics state, never replace image contents or relax audit.
    for warmup in range(32):
     frames=[v.get_rgba() for v in self.views.values()]
     if all(f is not None and f.size and f[:,:,:3].std()>1. for f in frames):break
     world.render()
    else:raise RuntimeError('CAMERA_INITIAL_WARMUP_FAILED')
    self.initial_camera_warmup_renders=warmup
'''.rstrip()
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("m['wrist_pose_follow']=", "m['initial_camera_warmup_renders']=getattr(self,'initial_camera_warmup_renders',0)\n  m['wrist_pose_follow']=")
ast.parse(s);(R/'threecam_recorder_v2.py').write_text(s)
versions=R/'source_versions';versions.mkdir(exist_ok=True);changes={}
for f in [*R.glob('realpose_*_trial.py'),*R.glob('policy_*_trial.py')]:
 raw=f.read_bytes();before=hashlib.sha256(raw).hexdigest();(versions/(before+'.py')).write_bytes(raw)
 s=f.read_text().replace('from threecam_recorder import Recorder','from threecam_recorder_v2 import Recorder');f.write_text(s);raw=f.read_bytes();after=hashlib.sha256(raw).hexdigest();(versions/(after+'.py')).write_bytes(raw);changes[f.name]=dict(before=before,after=after)
f=R/'policy_threecam_recorder.py';s=f.read_text().replace('from threecam_recorder import Recorder','from threecam_recorder_v2 import Recorder');f.write_text(s)
f=R/'verify_threecam.py';s=f.read_text();old="meta['threecam_recorder_sha256']==hashlib.sha256((R/'threecam_recorder.py').read_bytes()).hexdigest()";new="meta['threecam_recorder_sha256'] in {hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['threecam_recorder.py','threecam_recorder_v2.py']}";assert old in s;s=s.replace(old,new);f.write_text(s)
for name in ['threecam_recorder.py','threecam_recorder_v2.py','policy_threecam_recorder.py']:
 raw=(R/name).read_bytes();(versions/(hashlib.sha256(raw).hexdigest()+'.py')).write_bytes(raw)
# Expand only offline IK search radius; keep joint bounds and all acceptance gates.
s=(R/'posture_anchor.py').read_text().replace('radius=np.radians(10.)','radius=np.radians(20.)').replace('max_nfev=100','max_nfev=200');(R/'wide_posture_planner.py').write_text(s)
f=R/'plan_wide_threecam.py';s=f.read_text().replace('from posture_anchor import solve_posture','from wide_posture_planner import solve_posture').replace("group_id=f'{a}_wide_p{args.index:04d}',success=False","group_id=f'{a}_wide_p{args.index:04d}',planner_version='20deg_local_search_v2',success=False")
s=s.replace(' return qq',"\n mirror=max(abs(np.degrees(qq[f'idx{61+i}_arm_r_joint{i+1}']-sg*qq[f'idx{21+i}_arm_l_joint{i+1}'])) for i,sg in enumerate([-1,1,-1,1,-1,1,-1]))\n if mirror>15:raise ValueError('PLANNED_POSTURE_MIRROR_EXCEEDS15DEG')\n return qq")
ast.parse(s);f.write_text(s)
(W/'capture_and_planner_revision.json').write_text(json.dumps(dict(changes=changes,recorder_v2_sha256=hashlib.sha256((R/'threecam_recorder_v2.py').read_bytes()).hexdigest(),reason='Observed isolated first-frame black wrist image; same-state adaptive sensor warmup. Larger-position IK radius20deg with unchanged limits/collision/physical gates.'),indent=2))
print('VERSIONED_CAPTURE_AND_WIDE_IK_UPDATED')
