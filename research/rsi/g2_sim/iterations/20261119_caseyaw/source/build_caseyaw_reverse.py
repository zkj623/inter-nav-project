from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'plan_caseyaw.py').read_text().replace('xx=np.array([1.,0,0])','xx=np.array([-1.,0,0])').replace('for yaw in (90,):','for yaw in (90,270,0,180):').replace("out=ROOT/'caseyaw_raw_plans'","out=ROOT/'caseyaw_reverse_raw_plans'")
s=s.replace("'case_handheld':(.65,.50,.78)","'case_handheld':(.78,.65,.50)")
assert 'xx=np.array([-1.,0,0])' in s
ast.parse(s);(R/'plan_caseyaw_reverse.py').write_text(s)
s=(R/'connect_caseyaw.py').read_text().replace("R/'caseyaw_raw_plans/case_handheld.json'","R/'caseyaw_reverse_raw_plans/case_handheld.json'").replace("R/'caseyaw_plans'","R/'caseyaw_reverse_plans'").replace("f'caseyaw_planning_{a}.json'","f'caseyaw_reverse_planning_{a}.json'")
(R/'connect_caseyaw_reverse.py').write_text(s)
