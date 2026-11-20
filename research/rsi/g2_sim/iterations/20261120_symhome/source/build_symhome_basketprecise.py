from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'symhome_v4_basket_rectangular_trial.py').read_text().replace("ROOT/'symhome_v4_trials'","ROOT/'symhome_v4precise_trials'")
s=s.replace('CreateSolverPositionIterationCountAttr(64)','CreateSolverPositionIterationCountAttr(128)').replace('CreateSolverVelocityIterationCountAttr(16)','CreateSolverVelocityIterationCountAttr(32)').replace('64 position / 16 velocity','128 position / 32 velocity')
ast.parse(s);(R/'symhome_v4precise_basket_rectangular_trial.py').write_text(s)
