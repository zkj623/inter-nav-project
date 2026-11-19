from pathlib import Path
import json
R=Path(__file__).resolve().parent
for aid in ['bin_round_lidded','bin_lidded','bin_round_pedal']:
 p=json.loads((R/'liftall_validated_plans'/(aid+'.json')).read_text())['plans'][0]
 print(aid,json.dumps(p['candidate'],indent=2),flush=True)
