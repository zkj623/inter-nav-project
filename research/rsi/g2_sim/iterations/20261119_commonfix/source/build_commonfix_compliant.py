from pathlib import Path
import ast,json,copy
R=Path(__file__).resolve().parent;s=(R/'commonfix_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_trials'","ROOT/'commonfix_compliant_trials'")
s=s.replace("1000. if 'gripper' not in name else 20.","600. if 'gripper' not in name else 15.")
s=s.replace("100. if 'gripper' not in name else 1.","80. if 'gripper' not in name else 2.")
s=s.replace("20. if 'gripper' in n else 100000. if 'body_joint' in n else 1000.","15. if 'gripper' in n else 100000. if 'body_joint' in n else 600.")
s=s.replace("1. if 'gripper' in n else 5000. if 'body_joint' in n else 100.","2. if 'gripper' in n else 5000. if 'body_joint' in n else 80.")
s=s.replace("result['feedback_revision']='","result['feedback_revision']='compliant arm kp600 kd80, gripper kp15 kd2; original effort caps retained; ")
ast.parse(s);(R/'commonfix_compliant_case_handheld_trial.py').write_text(s)
folder=R/'commonfix_compliant_plans';folder.mkdir(exist_ok=True);p=json.loads((R/'commonfix_plans/case_handheld.json').read_text())['plans'][0];old=json.loads((R/'commonstart_plans/case_handheld.json').read_text())['plans'][0];(folder/'case_handheld.json').write_text(json.dumps(dict(asset='case_handheld',plans=[p,old]),indent=2))
