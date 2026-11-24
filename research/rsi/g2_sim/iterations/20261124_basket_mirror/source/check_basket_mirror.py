from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
rows=[]
for variant in ['mirror_right','mirror_home_seed']:
 p=json.loads((R/variant/'basket_rectangular.json').read_text())['plans'][0];a='basket_rectangular';plans[a]=p
 row={'variant':variant,'approach':edge(a,p['q_start'],p['q_pre']),'grasp_lift':check(a,p)}
 row['right_grasp_deg']={n:float(np.degrees(v)) for n,v in p['q_grasp'].items() if 'arm_r_joint' in n};rows.append(row);print(row,flush=True)
(R/'mirror_geometry.json').write_text(json.dumps(rows,indent=2))
