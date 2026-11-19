from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'plan_caseyaw.py').read_text().replace('for yaw in (90,):','for yaw in (270,90,180,0):').replace('for theta in (35,20,50,65):','for theta in (75,65,55):')
s=s.replace('np.linspace(-.35,.35,141)','np.linspace(-.8,.8,321)').replace("'case_handheld':(.65,.50,.78)","'case_handheld':(.78,.65,.88)")
s=s.replace('for distance in (.12,.15,.09,.18):','for distance in (.15,.18,.12):').replace("out=ROOT/'caseyaw_raw_plans'","out=ROOT/'caseyaw_steep_raw_plans'")
ast.parse(s);(R/'plan_caseyaw_steep.py').write_text(s)
s=(R/'connect_caseyaw.py').read_text().replace("R/'caseyaw_raw_plans/case_handheld.json'","R/'caseyaw_steep_raw_plans/case_handheld.json'").replace("R/'caseyaw_plans'","R/'caseyaw_steep_plans'").replace("f'caseyaw_planning_{a}.json'","f'caseyaw_steep_planning_{a}.json'")
(R/'connect_caseyaw_steep.py').write_text(s)
