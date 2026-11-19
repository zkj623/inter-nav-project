from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;old=R.parent/'g2-liftpair-20261119';repo=R.parent/'g2-publish-20261118/research/rsi/g2_sim'
for n in ['generated','robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies','vendor']:
 if not (R/n).exists():(R/n).symlink_to((old/n).resolve(),target_is_directory=True)
for p in old.glob('*.py'):shutil.copy2(p,R/p.name)
c=json.loads((repo/'ground_lift_training_config.json').read_text());(R/'ground_lift_training_config.json').write_text(json.dumps(c,indent=2))
out=R/'commonstart_base_plans';out.mkdir(exist_ok=True)
for a,b in c['validated_simulation_baselines'].items():
 arc=repo/'iterations'/b['iteration'];doc=json.loads((arc/'fixtures'/b['plan_dir']/(a+'.json')).read_text());p=doc['plans'][b['candidate']]
 p['source_baseline']=b;(out/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=[p]),indent=2))
 src=arc/'source'/b['harness'];dst=R/('commonstart_'+a+'_trial.py');s=src.read_text()
 import re
 s=re.sub(r"OUT=ROOT/[^\n]+", "OUT=ROOT/'commonstart_trials'/args.plan_dir/args.asset/f'candidate{args.candidate}_repeat{args.repeat}'",s,count=1)
 needle="        segment(qpre,q_grasp_pre,12.,'descent')"
 assert needle in s
 s=s.replace(needle,"        last=qpre\n        for waypoint in plan['q_approach']:\n            segment(last,waypoint,1.,'common_approach');last=waypoint")
 s=s.replace("segment(q_grasp_pre,q_grasp_pre,3.,'descent_hold')","segment(q_grasp_pre,q_grasp_pre,2.,'pregrasp_hold')")
 s=s.replace("prepose initialization'","common arm/gripper initial pose; collision-audited approach'")
 dst.write_text(s)
print('PREPARED',len(c['validated_assets']))
