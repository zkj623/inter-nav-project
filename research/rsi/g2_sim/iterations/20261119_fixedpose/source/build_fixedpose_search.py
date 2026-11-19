from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'screen_fixedpose.py').read_text().replace('from ideal_constraints import Envelope','from ideal_constraints import Envelope\nfrom fixedpose_collision import check as collision_check')
s=s.replace("f'fixedpose_screen_{pid}.json'","f'fixedpose_search_{pid}.json'")
s=s.replace("best['source_baseline']=row['baseline'];break", "best['source_baseline']=row['baseline']\n  hit=collision_check(row['asset'],best)\n  if hit:\n   attempts[-1]['collision']=hit;best=None;continue\n  break")
assert 'collision_check(row' in s
(R/'search_fixedpose.py').write_text(s)
