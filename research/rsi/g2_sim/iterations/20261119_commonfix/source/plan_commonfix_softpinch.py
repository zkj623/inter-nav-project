from pathlib import Path
import json,copy
from liftall_collision import check
R=Path(__file__).resolve().parent;a='basket_rectangular';original=json.loads((R/'commonstart_plans'/f'{a}.json').read_text())['plans'][0];out=[]
for opening in [-.03,-.05,-.07]:
 p=copy.deepcopy(original);p['hold_gripper_q']=opening
 for q in [p['q_close']]+p['q_lift']:
  for n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']:q[n]=opening
 p['soft_pinch_master_target_rad']=opening;p['adaptation_note']='Reduce master closing target to reduce force and avoid proxy-contact excursions; actual force and visible witness checks retained.'
 bad=check(a,p);print(opening,bad,flush=True)
 if bad is None:out.append(p)
folder=R/'commonfix_pinch_plans';folder.mkdir(exist_ok=True);(folder/f'{a}.json').write_text(json.dumps(dict(asset=a,plans=out),indent=2))
