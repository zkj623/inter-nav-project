from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonstartrange_case_handheld_trial.py').read_text().replace("ROOT/'commonstart_range_trials'","ROOT/'commonstart_force_trials'")
s=s.replace('force_error=20.-total_load','force_error=24.-total_load').replace('np.radians(.6)*feedback_dt','np.radians(1.5)*feedback_dt')
ast.parse(s);(R/'commonstartforce_case_handheld_trial.py').write_text(s)
