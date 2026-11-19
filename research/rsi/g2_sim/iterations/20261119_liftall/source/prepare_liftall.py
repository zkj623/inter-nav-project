from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;A=R.parent/'g2-publish-20261118/research/rsi/g2_sim'
for version in ['20261119_fixedpose','20261119_marginpose']:
 for folder in ['source','fixtures']:
  for p in (A/'iterations'/version/folder).rglob('*'):
   if p.is_file():
    dest=R/p.relative_to(A/'iterations'/version/folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(A/'ground_lift_training_config.json',R/'ground_lift_training_config.json')
for p in (R/'marginpose_selected7_validated_plans').glob('*.json'):
 for q in json.loads(p.read_text())['plans']:
  print(p.stem,q['mode'],q.get('strategy'),q.get('hold_gripper_q'),q['object_position'],flush=True)
