from pathlib import Path
import ast
R=Path(__file__).resolve().parent;name='basket_rectangular'
s=(R/f'commonstartslow_{name}_trial.py').read_text().replace("ROOT/'commonstart_slow_trials'","ROOT/'commonstart_pinch_trials'")
b=(R/'liftfix_balance_trial.py').read_text()
init=b[b.index("    pinch_shift ="):b.index("    balance =")]
s=s.replace("    balance =",init+"    balance =",1)
feedback=b[b.index("            elif plan.get('mode')=='bimanual_top':"):b.index('            solution,pe,re=k.solve(side,target,actual,current_base,tcp)')]
s=s.replace('            solution,pe,re=k.solve(side,target,actual,current_base,tcp)',feedback+'            solution,pe,re=k.solve(side,target,actual,current_base,tcp)')
s=s.replace("achieved=all(sum(v for n,v in forces.items() if f'gripper_{side}_' in n)>.75 for side in ('l','r'))","achieved=four_contact(samples[-1],1.)")
s=s.replace("        result['adaptive_grip_extra_m']=grip_extra","        if plan.get('mode')=='bimanual_top' and stable<3:raise TrialRejected('FOUR_BRANCH_PRELOAD_NOT_ESTABLISHED')\n        result['adaptive_grip_extra_m']=grip_extra")
ast.parse(s);(R/f'commonstartpinch_{name}_trial.py').write_text(s)
