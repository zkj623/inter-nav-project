from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'plan_liftpair_geometry.py').read_text().replace("['case_handheld','box_reinforced']","['case_handheld']").replace("R/'liftfix_pedal_plans'","R/'liftpair_high_plans'").replace("R/'liftpair_geometry_plans'","R/'liftpair_higharc_plans'").replace("R/'liftpair_geometry_search.json'","R/'liftpair_higharc_search.json'")
(R/'plan_liftpair_higharc.py').write_text(s)
