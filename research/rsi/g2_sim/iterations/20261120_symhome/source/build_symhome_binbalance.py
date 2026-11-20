from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'symhome_v4_bin_round_lidded_trial.py').read_text().replace("ROOT/'symhome_v4_trials'","ROOT/'symhome_v4balanced_trials'")
source=(R/'symhome_v4_case_handheld_trial.py').read_text();a=source.index("    squeeze = {'l': 0., 'r': 0.}");b=source.index('    def four_contact(',a)
x=s.index("    squeeze = {'l': 0., 'r': 0.}");y=s.index('    def four_contact(',x);s=s[:x]+source[a:b]+s[y:]
s=s.replace("    result['contact_definition']=","    result['feedback_controller']='case-style bounded opposite tilt0.4 gain, max8deg and1deg/s; normalized wrist load balance0.4deg/s; force target28N/side, squeeze<=20mm; original four-force preload/settle/hold gates retained'\n    result['contact_definition']=")
ast.parse(s);(R/'symhome_v4balanced_bin_round_lidded_trial.py').write_text(s)
