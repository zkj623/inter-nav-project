from pathlib import Path
import json,copy,sys,numpy as np
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
a=sys.argv[1];home=json.loads((R/'commonstart_planning.json').read_text())['home']
if a=='case_handheld':path=R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_liftpair/fixtures/liftpair_geometry_plans/case_handheld.json'
else:path=R/'commonfix_raw_plans'/f'{a}.json'
doc=json.loads(path.read_text());accepted=[];records=[]
for index,old in enumerate(doc['plans']):
 p=copy.deepcopy(old);plans[a]=p;bad=check(a,p);way=None
 if bad is None:
  if edge(a,home,p['q_pre']):way=[p['q_pre']]
  else:
   q=p['q_pre'].copy();fk=k.fk(q,base);ok=True
   for side in ['l','r']:
    t=fk[f'arm_{side}_end_link'].copy();t[2,3]+=.15;q,pe,re=k.solve(side,t,q,base,np.eye(4))
    if pe>.002 or re>.02:ok=False;break
   if ok and edge(a,home,q) and edge(a,q,p['q_pre']):way=[q,p['q_pre']]
 if way:
  p.update(q_start=home,q_approach=way,source_geometry_candidate=index);accepted.append(p)
 records.append(dict(candidate=index,collision=bad,approach_pass=way is not None));print(a,records[-1],flush=True)
out=R/'commonfix_plans';out.mkdir(exist_ok=True);(out/f'{a}.json').write_text(json.dumps(dict(asset=a,plans=accepted),indent=2));(R/f'commonfix_planning_{a}.json').write_text(json.dumps(records,indent=2));print('DONE',a,len(accepted),flush=True)
