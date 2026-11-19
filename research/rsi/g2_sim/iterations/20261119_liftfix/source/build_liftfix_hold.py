from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'liftfix_preload_trial.py').read_text().replace("ROOT/'liftfix_preload_trials'","ROOT/'liftfix_hold_trials'")
s=s.replace('ready=four_contact(recent,4.) and velocity<.005','ready=four_contact(recent,1.) and velocity<.015')
s=s.replace('if settled_updates>=8:break','if settled_updates>=4:break').replace('if settled_updates<8:raise','if settled_updates<4:raise')
s=s.replace('settle >4N each, speed <5mm/s for 8 updates','restore original settle gate >1N each, speed <15mm/s for 4 updates; final acceptance unchanged')
ast.parse(s);(R/'liftfix_hold_trial.py').write_text(s)
