from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'caseyaw_level_trial.py').read_text().replace("ROOT/'caseyaw_level_trials'","ROOT/'caseyaw_damped_trials'")
s=s.replace('np.radians(1.5)*feedback_dt*imbalance','np.radians(.4)*feedback_dt*imbalance')
s=s.replace("for s in samples[-3:]])) for finger in ('inner','outer')]","for s in samples[-5:]])) for finger in ('inner','outer')]")
s=s.replace('force_error=24.-total_load','force_error=28.-total_load')
s=s.replace("ready=len(recent['diagnostic_contacts'])==4", "ready=four_contact(recent,2.) and len(recent['diagnostic_contacts'])==4")
s=s.replace('if settled_updates>=4:break','if settled_updates>=8:break').replace('if settled_updates<4:raise','if settled_updates<8:raise')
s=s.replace("    result['capture_gate']=", "    result['damped_feedback']='load balance0.4deg/s; load average0.5s; target28N/side; settle requires four forces>2N for8 updates; final acceptance unchanged'\n    result['capture_gate']=")
ast.parse(s);(R/'caseyaw_damped_trial.py').write_text(s)
