from pathlib import Path
import json
R=Path(__file__).resolve().parent
for variant,asset,pd,c in [('diverse','chair_dining','strategy15_diverse_validated_plans',0),('fixedfeature','stool_rectangular','strategy15_guard_validated_plans',1)]:
 p=R/f'strategy15_{variant}_trials'/pd/asset/f'candidate{c}_repeat1/result.json'
 d=json.loads(p.read_text())
 print(asset,d['failure'])
 for event in d.get('surface_error_events',[])[:2]:
  print({k:event[k] for k in ('t','phase','link')},'max_object_error',max(event['object_error_m']),'max_finger_error',max(event['finger_error_m']),'minimum_recorded_separation',min(x[0] if isinstance(x,list) else x for x in event['separation']))
