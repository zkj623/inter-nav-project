from pathlib import Path
import shutil,json,ast
R=Path(__file__).resolve().parent;old=R.parent/'g2-commonfix-20261119'
for n in ['generated','robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies','vendor']:
 if not (R/n).exists():(R/n).symlink_to((old/n).resolve(),target_is_directory=True)
for pat in ['*.py','*.json']:
 for p in old.glob(pat):shutil.copy2(p,R/p.name)
for p in old.iterdir():
 if p.is_dir() and p.name.endswith('plans'):shutil.copytree(p,R/p.name,dirs_exist_ok=True)
s=(R/'plan_liftfix_pedal.py').read_text().replace('for yaw in (0,90):','for yaw in (90,):').replace('for distance in (.06,.09,.03,.12,0.,-.03):','for distance in (.12,.15,.09,.18):').replace("out=ROOT/'liftfix_pedal_plans'","out=ROOT/'caseyaw_raw_plans'")
assert 'for yaw in (90,):' in s
ast.parse(s);(R/'plan_caseyaw.py').write_text(s)
s=(R/'plan_commonfix.py').read_text().replace("path=R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_liftpair/fixtures/liftpair_geometry_plans/case_handheld.json'","path=R/'caseyaw_raw_plans/case_handheld.json'").replace("R/'commonfix_plans'","R/'caseyaw_plans'").replace("f'commonfix_planning_{a}.json'","f'caseyaw_planning_{a}.json'")
(R/'connect_caseyaw.py').write_text(s)
s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'caseyaw_trials'")
(R/'caseyaw_trial.py').write_text(s)
print('YAW90_PREPARED',flush=True)
