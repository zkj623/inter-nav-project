from pathlib import Path
import ast
R=Path(__file__).resolve().parent
for label,opening in [('narrow',-.4),('wide',-.75)]:
 s=(R/'plan_liftfix_pedal.py').read_text().replace('openq=-.6;',f'openq={opening};').replace("out=ROOT/'liftfix_pedal_plans'",f"out=ROOT/'commonfix_{label}_raw_plans'")
 ast.parse(s);(R/f'plan_commonfix_{label}.py').write_text(s)
 s=(R/'plan_commonfix.py').read_text().replace("path=R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_liftpair/fixtures/liftpair_geometry_plans/case_handheld.json'",f"path=R/'commonfix_{label}_raw_plans/case_handheld.json'")
 s=s.replace("R/'commonfix_plans'",f"R/'commonfix_{label}_plans'").replace("f'commonfix_planning_{a}.json'",f"f'commonfix_{label}_planning_{{a}}.json'")
 (R/f'connect_commonfix_{label}.py').write_text(s)
