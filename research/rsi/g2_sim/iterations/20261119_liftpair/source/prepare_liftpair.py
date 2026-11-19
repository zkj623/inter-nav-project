from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;old=R.parent/'g2-liftfix-20261119'
for name in ['generated','robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies','vendor']:
 p=R/name
 if not p.exists():p.symlink_to((old/name).resolve(),target_is_directory=True)
for pattern in ['*.py','*.json']:
 for p in old.glob(pattern):shutil.copy2(p,R/p.name)
for p in old.iterdir():
 if p.is_dir() and p.name.endswith('plans'):shutil.copytree(p,R/p.name,dirs_exist_ok=True)
shutil.copy2(R.parent/'g2-publish-20261118/research/rsi/g2_sim/ground_lift_training_config.json',R/'ground_lift_training_config.json')
for aid,rel in [('case_handheld','liftfix_hold_trials/liftfix_arc_plans/case_handheld/candidate1_repeat1'),('box_reinforced','liftfix_balance_trials/liftfix_box_plans/box_reinforced/candidate0_repeat1')]:
 ss=json.loads((old/rel/'samples.json').read_text());print(aid,'SAMPLE_KEYS',list(ss[0]))
 for s in [s for s in ss if s['phase']=='hold']:
  print(round(s['t'],3),round(s['tilt_deg'],2),[round(s['finger_forces'].get(f'gripper_{side}_{f}_link4',0.),2) for side in ['l','r'] for f in ['inner','outer']])
