from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_holdangle_trials'")
s=s.replace("1000. if 'gripper' not in name else 20.","1000. if 'gripper' not in name else 80.").replace("100. if 'gripper' not in name else 1.","100. if 'gripper' not in name else 2.")
s=s.replace("20. if 'gripper' in n else 100000. if 'body_joint' in n else 1000.","80. if 'gripper' in n else 100000. if 'body_joint' in n else 1000.").replace("1. if 'gripper' in n else 5000. if 'body_joint' in n else 100.","2. if 'gripper' in n else 5000. if 'body_joint' in n else 100.")
s=s.replace("result['constraint_accuracy']=","result['gripper_gain_revision']='master kp80 kd2, original URDF effort and velocity limits retained; target aperture held more accurately under load'\n    result['constraint_accuracy']=")
ast.parse(s);(R/'commonfix_holdangle_case_handheld_trial.py').write_text(s)
