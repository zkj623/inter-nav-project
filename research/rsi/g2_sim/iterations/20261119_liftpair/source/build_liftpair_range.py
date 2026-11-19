from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'liftpair_soft_trial.py').read_text().replace("ROOT/'liftpair_soft_trials'","ROOT/'liftpair_range_trials'")
old='squeeze[side]=float(np.clip(squeeze[side]+rate*feedback_dt,0.,.012))'
assert old in s;s=s.replace(old,'squeeze[side]=float(np.clip(squeeze[side]+rate*feedback_dt,0.,.020))')
s=s.replace("result['feedback_revision']='", "result['feedback_revision']='bounded squeeze expanded from 12 to 20mm; original motor limits and >35N release retained; ")
ast.parse(s);(R/'liftpair_range_trial.py').write_text(s)
