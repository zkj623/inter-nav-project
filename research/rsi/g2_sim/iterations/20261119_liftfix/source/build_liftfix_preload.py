from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'liftfix_balance_trial.py').read_text().replace("ROOT/'liftfix_balance_trials'","ROOT/'liftfix_preload_trials'")
s=s.replace('if min(loads)<2. and max(loads)<30.:','if min(loads)<5. and max(loads)<30.:')
s=s.replace('if min(loads)<2. and max(loads)>4.:','if min(loads)<5. and max(loads)>6.:')
s=s.replace("achieved=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>1.","achieved=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>4.")
s=s.replace('ready=four_contact(recent,1.) and velocity<.015','ready=four_contact(recent,4.) and velocity<.005')
s=s.replace('if settled_updates>=4:break','if settled_updates>=8:break').replace("if settled_updates<4:raise", "if settled_updates<8:raise")
s=s.replace("result['feedback_revision']='", "result['feedback_revision']='Stronger spread preload: target 5N per distal finger; settle >4N each, speed <5mm/s for 8 updates; ")
ast.parse(s);(R/'liftfix_preload_trial.py').write_text(s)
