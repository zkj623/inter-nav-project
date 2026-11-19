from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;old=R.parent/'g2-commonstart-20261119';repo=R.parent/'g2-publish-20261118/research/rsi/g2_sim'
for n in ['generated','robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies','vendor']:
 if not (R/n).exists():(R/n).symlink_to((old/n).resolve(),target_is_directory=True)
for pat in ['*.py','*.json']:
 for p in old.glob(pat):shutil.copy2(p,R/p.name)
for p in old.iterdir():
 if p.is_dir() and p.name.endswith('plans'):shutil.copytree(p,R/p.name,dirs_exist_ok=True)
shutil.copy2(repo/'ground_lift_training_config.json',R/'ground_lift_training_config.json')
for a,src in [('basket_rectangular','commonstartslow_basket_rectangular_trial.py'),('case_handheld','commonstartforce_case_handheld_trial.py')]:
 s=(R/src).read_text();import re
 s=re.sub(r"OUT=ROOT/[^\n]+","OUT=ROOT/'commonfix_trials'/args.plan_dir/args.asset/f'candidate{args.candidate}_repeat{args.repeat}'",s,count=1)
 s=s.replace("if gap>.0015:failure=failure or 'LOADED_VISIBLE_SURFACE_GAP'","if gap>.0015:\n                    result.setdefault('loaded_gap_events',[]).append(dict(t=step/120.,phase=phase,link=n,gap_m=gap,force_n=float(forces[i])))\n                    failure=failure or 'LOADED_VISIBLE_SURFACE_GAP'")
 (R/f'commonfix_{a}_trial.py').write_text(s)
# Frame/rim grasp generator, already fixed to the same body posture.
s=(R/'plan_liftfix_feature.py').read_text().replace("out=ROOT/'liftfix_feature_plans'","out=ROOT/'commonfix_raw_plans'")
s=s.replace('max_plans=3','max_plans=6')
(R/'plan_commonfix_basket.py').write_text(s)
print('PREPARED')
