from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'verify_fixedpose_geometry.py').read_text().replace("'fixedpose_*_trials/**/result.json'","'marginpose_*_trials/**/result.json'").replace("'fixedpose_physical_geometry_review.json'","'marginpose_physical_geometry_review.json'")
ast.parse(s);(R/'verify_marginpose_geometry.py').write_text(s)
