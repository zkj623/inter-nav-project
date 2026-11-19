from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;old=R.parent/'g2-liftall-20261119'
for name in ['generated','robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies','vendor']:
 p=R/name
 if not p.exists():p.symlink_to((old/name).resolve(),target_is_directory=True)
for p in old.glob('*.py'):shutil.copy2(p,R/p.name)
for p in old.glob('*.json'):shutil.copy2(p,R/p.name)
for p in old.iterdir():
 if p.is_dir() and p.name.endswith('plans'):shutil.copytree(p,R/p.name,dirs_exist_ok=True)
shutil.copy2(R.parent/'g2-publish-20261118/research/rsi/g2_sim/ground_lift_training_config.json',R/'ground_lift_training_config.json')
for aid in ['basket_rectangular','basket_woven','bin_round_pedal','box_reinforced','case_handheld','bin_pedal']:
 print('\nASSET',aid)
 for folder in ['liftall_validated_plans','strategy15_frame_validated_plans','strategy15_tilt_validated_plans','strategy15_retry_validated_plans','strategy15_validated_plans','catalog_dual_validated_plans']:
  f=R/folder/(aid+'.json')
  if f.exists():
   ps=json.loads(f.read_text())['plans'];print(folder,len(ps),[(p.get('mode'),p.get('strategy')) for p in ps[:2]])
 p=R/'liftall_validated_plans'/(aid+'.json')
 if p.exists():
  q=json.loads(p.read_text())['plans'][0];print('CURRENT',q['candidate'],'tcp',q['tcp'],'gripper',q['q_close'].get('idx31_gripper_l_inner_joint1'))
