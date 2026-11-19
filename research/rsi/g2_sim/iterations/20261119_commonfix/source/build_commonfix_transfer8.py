from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_transfer_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_transfer_trials'","ROOT/'commonfix_transfer8_trials'")
s=s.replace("if n.startswith(f'gripper_{side}_'))>2.","if n.startswith(f'gripper_{side}_'))>8.")
s=s.replace('np.array([.02,0.,.01])','np.array([.04,0.,.01])').replace("result['load_transfer_m']=[.02,0.,.01]","result['load_transfer_m']=[.04,0.,.01]").replace('first bilateral contact >2N each','first bilateral contact >8N each')
ast.parse(s);(R/'commonfix_transfer8_case_handheld_trial.py').write_text(s)
