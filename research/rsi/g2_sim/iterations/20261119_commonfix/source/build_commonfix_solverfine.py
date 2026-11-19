from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_solverfine_trials'")
s=s.replace('CreateSolverPositionIterationCountAttr(64)','CreateSolverPositionIterationCountAttr(256)').replace('CreateSolverVelocityIterationCountAttr(16)','CreateSolverVelocityIterationCountAttr(64)').replace('solver 64 position / 16 velocity','solver 256 position / 64 velocity')
ast.parse(s);(R/'commonfix_solverfine_case_handheld_trial.py').write_text(s)
