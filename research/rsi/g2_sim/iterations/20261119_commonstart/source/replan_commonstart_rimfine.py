from pathlib import Path
import copy,json,numpy as np
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
a='basket_rectangular';original=copy.deepcopy(plans[a]);home=json.loads((R/'commonstart_planning.json').read_text())['home'];out=[]
for dx,dz in [(-.01,.005),(-.02,0.)]:
 p=copy.deepcopy(original);qs=[p['q_pre'],p['q_grasp'],p['q_close']]+p['q_lift'];mapped=[];bad=False
 for q in qs:
  fk=k.fk(q,base);new=q.copy()
  for side in ['l','r']:
   target=fk[f'arm_{side}_end_link'].copy();target[:3,3]+=[dx,0,dz];new,pe,re=k.solve(side,target,new,base,np.eye(4))
   if pe>.002 or re>.02:bad=True;break
  if bad:break
  mapped.append(new)
 if bad:continue
 p.update(q_pre=mapped[0],q_grasp=mapped[1],q_close=mapped[2],q_lift=mapped[3:]);p['object_position'][0]+=dx;plans[a]=p
 if check(a,p) or not edge(a,home,p['q_pre']):continue
 p.update(q_start=home,q_approach=[p['q_pre']],rim_fine_adjustment_m=[dx,0,dz]);out.append(p)
folder=R/'commonstart_rimfine_plans';folder.mkdir(exist_ok=True);(folder/(a+'.json')).write_text(json.dumps(dict(asset=a,plans=out),indent=2));print('FINE_RIM',len(out),flush=True)
