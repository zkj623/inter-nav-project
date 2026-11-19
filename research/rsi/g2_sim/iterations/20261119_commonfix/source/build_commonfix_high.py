from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'plan_commonfix.py').read_text().replace('liftpair_geometry_plans/case_handheld.json','liftpair_higharc_plans/case_handheld.json').replace("R/'commonfix_plans'","R/'commonfix_high_plans'").replace("f'commonfix_planning_{a}.json'","f'commonfix_highplanning_{a}.json'")
(R/'plan_commonfix_high.py').write_text(s)
