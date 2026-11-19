from pathlib import Path
import json,copy
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
a='basket_rectangular';doc=json.loads((R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_liftfix/fixtures/liftfix_plans/basket_rectangular.json').read_text());home=json.loads((R/'commonstart_planning.json').read_text())['home'];accepted=[]
for index in [1,2]:
 p=copy.deepcopy(doc['plans'][index]);plans[a]=p
 if check(a,p) is not None:continue
 if not edge(a,home,p['q_pre']):continue
 p.update(q_start=home,q_approach=[p['q_pre']],common_start_scope='identical common q_start; newly selected rim depth; audited full approach',rim_source_candidate=index);accepted.append(p)
out=R/'commonstart_rim_plans';out.mkdir(exist_ok=True);(out/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=accepted),indent=2));print('RIM_PLANS',len(accepted),flush=True)
