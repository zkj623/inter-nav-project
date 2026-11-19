from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'strategy15_frame_candidates.py').read_text().replace("('table_side','stool_wood_four_leg','stool_rectangular')","('table_side','stool_wood_four_leg','stool_rectangular','case_handheld')").replace("'case_handheld':(.45,.92)","'case_handheld':(.80,1.)")
ast.parse(s);(R/'commonfix_edge_candidates.py').write_text(s)
s=(R/'plan_liftfix_feature.py').read_text().replace('from strategy15_frame_candidates import make_candidates','from commonfix_edge_candidates import make_candidates').replace("out=ROOT/'liftfix_feature_plans'","out=ROOT/'commonfix_edge_raw_plans'")
ast.parse(s);(R/'plan_commonfix_edge.py').write_text(s)
s=(R/'plan_commonfix.py').read_text().replace("path=R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_liftpair/fixtures/liftpair_geometry_plans/case_handheld.json'","path=R/'commonfix_edge_raw_plans/case_handheld.json'").replace("R/'commonfix_plans'","R/'commonfix_edge_plans'").replace("f'commonfix_planning_{a}.json'","f'commonfix_edgeplanning_{a}.json'")
(R/'connect_commonfix_edge.py').write_text(s)
