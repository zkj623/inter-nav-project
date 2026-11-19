from pathlib import Path
import ast,json,os,signal
R=Path(__file__).resolve().parent
for pid in [2112537,2112538]:
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  cwd=Path(f'/proc/{pid}/cwd').resolve()
  if cwd==R and ('fixedpose_side_trial.py' in cmd or 'fixedpose_support_trial.py' in cmd):os.kill(pid,signal.SIGTERM)
 except FileNotFoundError:pass
for label in ['side','deadband','support']:
 s=(R/f'fixedpose_{label}_trial.py').read_text().replace("rec['joint_positions']=qd","rec['joint_positions']={n:float(v) for n,v in qd.items()}")
 s=s.replace(f"'fixedpose_{label}_trials'",f"'fixedpose_{label}v2_trials'")
 ast.parse(s);(R/f'fixedpose_{label}v2_trial.py').write_text(s)
(R/'startup_recording_errors.json').write_text(json.dumps(dict(reason='Added joint-position logging used numpy float32, JSON serialization failed; corrected to Python float in v2',cancelled_pids=[2112537,2112538],invalid_pose_trial='fixedpose_side_trials/fixedpose_0_validated_plans/box_reinforced/candidate0_repeat90'),indent=2))
s=(R/'search_fixedpose.py').read_text().replace('pid=int(sys.argv[1]);','profiles.append([-52.5,150,-82.5,0,0])\npid=int(sys.argv[1]);')
(R/'search_fixedpose_mid.py').write_text(s)
