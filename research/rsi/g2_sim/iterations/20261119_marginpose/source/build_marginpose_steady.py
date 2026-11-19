from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'marginpose_deadband_trial.py').read_text()
s=s.replace("'marginpose_deadband_trials'","'marginpose_steady_trials'")
old="loads=[latest.get(f'gripper_{side}_{finger}_link4',0.) for finger in ('inner','outer')]"
new="loads=[float(np.mean([s['finger_forces'].get(f'gripper_{side}_{finger}_link4',0.) for s in samples[-3:]])) for finger in ('inner','outer')]"
assert old in s;s=s.replace(old,new).replace('for _ in range(30):','for _ in range(60):')
ast.parse(s);(R/'marginpose_steady_trial.py').write_text(s)
