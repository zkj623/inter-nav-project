from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'plan_liftfix_pedal.py').read_text().replace("'case_handheld':(.65,.50,.78)","'case_handheld':(.78,.88)").replace("out=ROOT/'liftfix_pedal_plans'","out=ROOT/'liftpair_high_plans'")
ast.parse(s);(R/'plan_liftpair_high.py').write_text(s)
