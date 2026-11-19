"""Check final evidence consistency and small settling motions of the frozen initial layouts."""
import argparse,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assets=set();used=set();motion=[];task_count=required=controls=0;scenes=[]
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());p=json.loads((ep.parent/'physics_smoke.json').read_text());assert p['passed'] and p['scene']==d['id'];assert len(p['objects'])==len(d['objects'])
 for o,r in zip(d['objects'],p['objects']):
  assert o['asset_id']==r['asset_id'];assets.add(o['asset_id']);delta=math.dist(o['position_m'][:2],r['final_position'][:2]);motion.append(delta)
  assert delta<.0125,(d['id'],o['instance_id'],'settled center shifts by at least half a grid cell',delta)
 for t in d['tasks']:
  assert t['cleared_route_length_m']>=5.0-1e-9,t['id']
  task_count+=1;required+=t['class']=='interaction_required';controls+=t['class']=='no_interaction_optimal';used.update(d['objects'][j]['asset_id'] for j in t.get('optimistic_relocation_subset',[]))
 scenes.append(d['id'])
assert len(scenes)==10 and task_count==100 and required==80 and controls==20 and len(assets)==15 and used==assets
for name in ('geometry_validation.json','layout_validation.json','diversity_validation.json','physics_validation.json','usd_validation.json','rematerialization_validation.json'):
 assert json.loads((a.out/name).read_text())['passed'],name
assert json.loads((a.out/'floor_support_validation.json').read_text())['passed']
report={'passed':True,'revision':d['revision'],'scenes':10,'fixed_layouts':10,'tasks':100,'interaction_required':80,'no_interaction_optimal':20,'object_types_present':15,'object_types_in_selected_clearing_plans':15,'minimum_reference_route_m':5.0,'maximum_settled_planar_shift_from_authored_position_m':max(motion),'settled_centers_within_half_grid_cell':True,'scope':'static raster task evidence, initial passive settling, USD loading and same-machine separate-directory reconstruction; G2 robot rollouts and parked-state settling remain untested'}
(a.out/'verification_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
